.PHONY: build run clean

build:
	docker build -f docker/Dockerfile -t intro-detector .

run:
	docker run -it --rm \
	  --gpus all \
	  -v $(PWD)/data:/data \
	  intro-detector

run-cpu:
	docker run -it --rm \
	  -v $(PWD)/data:/data \
	  intro-detector

compose:
	docker-compose up --build

clean:
	find data -name "*_preprocessed.mp4" -delete
	rm -f data/results.json