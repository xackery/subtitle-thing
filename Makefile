
build:
	docker build -t systran-faster-whisper:latest .
vod-%:
	docker run --rm --gpus all \
		-v "$$PWD:/data" \
		systran-faster-whisper:latest \
		/data/$*.mp4 \
		--model-size large-v3 \
		--device cuda \
		--compute-type float16 \
		--language en \
		--output-dir /data
