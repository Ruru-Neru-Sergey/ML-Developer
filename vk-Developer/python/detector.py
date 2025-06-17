import os
import subprocess
import json
import cv2
import tempfile
import shutil
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from .feature_extractor import FeatureExtractor
from .utils import timeit, load_config, seconds_to_time_str, setup_logger 

class IntroDetector:
    def download_and_process(self, json_path, output_path):
        """Скачивает видео из JSON и обрабатывает их"""
        # Создаем временную директорию для видео
        temp_dir = tempfile.mkdtemp()
    
        # Скачиваем видео
        download_script = os.path.join(os.path.dirname(__file__), 'download_videos.py')
        subprocess.run([
        'python', download_script,
        '--json', json_path,
        '--output', temp_dir
        ], check=True)
    
        # Обрабатываем скачанные видео
        results = self.process_directory(temp_dir, output_path)
    
        # Очистка временных файлов
        shutil.rmtree(temp_dir)
    
        return results
    def __init__(self, config):
        self.config = config
        self.extractor = FeatureExtractor()
        self.logger = setup_logger()
        self.similarity_threshold = config.get('similarity_threshold', 0.82)
        self.min_intro_length = config.get('min_intro_length', 3)
        self.max_intro_length = config.get('max_intro_length', 30)
    
    @timeit
    def preprocess_video(self, input_path):
        """Выполняет предобработку видео с помощью Go-бинарника"""
        try:
            cmd = ['go-preprocessor', input_path]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            output_path = result.stdout.strip()
            self.logger.info(f"Preprocessed video saved to: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Preprocessing failed: {e.stderr}")
            return None

    @timeit
    def find_intro(self, video_path):
        """Основная функция поиска заставки"""
        # Предобработка видео
        processed_path = self.preprocess_video(video_path)
        if not processed_path or not os.path.exists(processed_path):
            self.logger.error(f"Preprocessing failed for {video_path}")
            return None

        cap = cv2.VideoCapture(processed_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Параметры выборки
        sample_rate = max(1, int(fps // self.config['frames_per_second']))
        frames = []
        frame_times = []
        
        # Чтение и выборка кадров
        self.logger.info(f"Processing {os.path.basename(video_path)}...")
        for i in tqdm(range(total_frames), desc=f"Reading frames"):
            ret, frame = cap.read()
            if not ret:
                break
                
            if i % sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_times.append(i / fps)
        
        cap.release()
        
        if not frames:
            self.logger.warning("No frames extracted")
            return None
        
        # Извлечение признаков
        features = []
        batch_size = self.config['batch_size']
        for i in tqdm(range(0, len(frames), batch_size, desc="Extracting features")):
            batch = frames[i:i+batch_size]
            features_batch = self.extractor.batch_extract(batch)
            features.append(features_batch if features_batch.ndim > 1 else [features_batch])
        
        features = np.array(features)
        
        # Поиск последовательностей
        similarity = self.calculate_similarity(features)
        min_distance = int(fps * self.config['min_intro_length'] / 2)
        peaks, _ = find_peaks(  # Исправлен вызов find_peaks
            similarity, 
            height=self.similarity_threshold, 
            distance=min_distance
        )
        
        # Анализ результатов
        intro_candidates = self.analyze_peaks(peaks, frame_times, fps)
        best_candidate = self.select_best_candidate(intro_candidates)
        
        # Очистка временных файлов
        if self.config.get('cleanup', True):
            os.remove(processed_path)
        
        return {
            "video": os.path.basename(video_path),
            "processed": os.path.basename(processed_path),
            "result": best_candidate,
            "frames_processed": len(frames),
            "features_shape": features.shape,
            "peaks_found": len(peaks)
        }
    
    def calculate_similarity(self, features):
        """Вычисляет матрицу схожести кадров"""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = features / norms
        
        # Вычисляем попарную косинусную схожесть
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Усредняем схожесть для каждого кадра с его соседями
        window_size = min(5, len(normalized))
        similarity = np.zeros(len(normalized))
        
        for i in range(len(normalized)):
            start = max(0, i - window_size)
            end = min(len(normalized), i + window_size + 1)
            similarity[i] = np.mean(similarity_matrix[i, start:end])
        
        return similarity
    
    def analyze_peaks(self, peaks, frame_times, fps):
        """Анализирует группы пиков"""
        if not peaks:
            return []
        
        # Группировка последовательных пиков
        groups = []
        current_group = [peaks[0]]
        
        for i in range(1, len(peaks)):
            if peaks[i] - current_group[-1] <= fps * 2:  # 2-секундный порог
                current_group.append(peaks[i])
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [peaks[i]]
        
        if len(current_group) > 1:
            groups.append(current_group)
        
        # Формирование кандидатов
        candidates = []
        min_length = self.config['min_intro_length']
        max_length = self.config['max_intro_length']
        
        for group in groups:
            start_idx = group[0]
            end_idx = group[-1]
            start_time = frame_times[start_idx]
            end_time = frame_times[end_idx]
            duration = end_time - start_time
            
            if min_length <= duration <= max_length:
                confidence = len(group) / duration
                candidates.append({
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                    "confidence": confidence,
                    "peak_count": len(group)
                })
        
        return candidates
    
    def select_best_candidate(self, candidates):
        """Выбирает лучшего кандидата"""
        if not candidates:
            return {"start": 0, "end": 0, "confidence": 0}
        
        # Выбираем кандидата с наибольшей уверенностью
        return max(candidates, key=lambda x: x["confidence"])
    
    @timeit
    def process_directory(self, input_dir, output_path):
        """Обрабатывает все видео в директории"""
        results = {}
        video_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]
        
        if not video_files:
            raise FileNotFoundError(f"No video files found in {input_dir}")
        
        self.logger.info(f"Found {len(video_files)} videos to process")
        
        for filename in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(input_dir, filename)
            video_id = os.path.splitext(filename)[0]
            
            try:
                detection_result = self.find_intro(video_path)
                if detection_result is None:
                    self.logger.warning(f"No intro detected in {filename}")
                    results[video_id] = {
                        "url": f"https://vkvideo.ru/video{video_id}",
                        "name": filename,
                        "start": "0:00:00",
                        "end": "0:00:00"
                    }
                    continue
                
                # Преобразование секунд в формат ЧЧ:ММ:СС
                start_time = seconds_to_time_str(detection_result['result']['start'])
                end_time = seconds_to_time_str(detection_result['result']['end'])
                
                # Формируем результат в требуемом формате
                results[video_id] = {
                    "url": f"https://vkvideo.ru/video{video_id}",
                    "name": filename,
                    "start": start_time,
                    "end": end_time
                }
                
            except Exception as e:
                self.logger.error(f"Failed for {filename}: {str(e)}")
                results[video_id] = {
                    "url": f"https://vkvideo.ru/video{video_id}",
                    "name": filename,
                    "start": "0:00:00",
                    "end": "0:00:00",
                    "error": str(e)
                }
        
        # Сохраняем результаты в JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_path}")
        return results

if __name__ == "__main__":
    import argparse
    from .utils import load_config
    
    parser = argparse.ArgumentParser(description='Detect TV show intros')
    parser.add_argument('--mode', choices=['local', 'web'], required=True, help='Processing mode')
    parser.add_argument('--input', type=str, help='Path to video directory or JSON file')
    parser.add_argument('--output', type=str, default='intro_results.json', help='Output JSON path')
    args = parser.parse_args()
    
    config = load_config()
    detector = IntroDetector(config)
    
    try:
        if args.mode == 'local':
            print(f"Starting intro detection on local videos in: {args.input}")
            results = detector.process_directory(args.input, args.output)
        elif args.mode == 'web':
            print(f"Starting intro detection on web videos from: {args.input}")
            results = detector.download_and_process(args.input, args.output)
        
        print(f"Processing complete! Results saved to {args.output}")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        raise