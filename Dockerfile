FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Face detector model (OpenCV Zoo, YuNet, MIT licence)
RUN curl -sSL -o face_detection_yunet_2023mar.onnx \
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

COPY app.py pipeline.py ./

ENV PORT=10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
