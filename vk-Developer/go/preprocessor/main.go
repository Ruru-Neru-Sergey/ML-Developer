package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go-preprocessor <input_path> [output_path]")
	}

	inputPath := os.Args[1]
	outputPath := ""

	if len(os.Args) > 2 {
		outputPath = os.Args[2]
	} else {
		// Генерация имени выходного файла
		ext := filepath.Ext(inputPath)
		base := strings.TrimSuffix(filepath.Base(inputPath), ext)
		outputPath = filepath.Join(filepath.Dir(inputPath), base+"_preprocessed.mp4")
	}

	// Создаем выходную директорию
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		log.Fatalf("Error creating directory: %v", err)
	}

	// Команда ffmpeg для конвертации
	cmd := exec.Command("ffmpeg",
		"-i", inputPath,
		"-c:v", "libx264",
		"-preset", "fast",
		"-crf", "23",
		"-vf", "scale=640:360", // уменьшаем разрешение для ускорения обработки
		"-an", // удаляем аудио
		"-y",  // перезаписать без подтверждения
		outputPath)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		log.Fatalf("FFmpeg processing failed: %v", err)
	}

	fmt.Println(outputPath) // Возвращаем путь к обработанному файлу
}