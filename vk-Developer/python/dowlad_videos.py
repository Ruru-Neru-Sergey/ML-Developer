import os
import requests
import json
from tqdm import tqdm
from urllib.parse import urlparse

def download_video(url, output_dir):
    """Скачивает видео по URL и сохраняет в указанную директорию"""
    try:
        # Извлекаем имя файла из URL
        parsed = urlparse(url)
        video_id = parsed.path.split('/')[-1]
        filename = f"{video_id}.mp4"
        output_path = os.path.join(output_dir, filename)
        
        # Проверяем, не скачано ли видео уже
        if os.path.exists(output_path):
            return filename
        
        # Скачиваем видео с прогресс-баром
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB блоки
        
        with open(output_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

def download_from_json(json_path, output_dir):
    """Скачивает все видео из JSON-файла"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    downloaded = {}
    
    for video_id, info in tqdm(data.items(), desc="Processing videos"):
        url = info['url']
        filename = download_video(url, output_dir)
        if filename:
            downloaded[video_id] = filename
    
    return downloaded

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download videos from JSON')
    parser.add_argument('--json', required=True, help='Path to JSON file')
    parser.add_argument('--output', required=True, help='Output directory for videos')
    args = parser.parse_args()
    
    print(f"Downloading videos from {args.json} to {args.output}")
    downloaded = download_from_json(args.json, args.output)
    print(f"Downloaded {len(downloaded)} videos")