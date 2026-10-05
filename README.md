# EyesOnly video worker

Turns a member's ≤10 s video into an eyes-only clip:

- finds both eyes in every frame (OpenCV YuNet), smooths the track,
- moves a crop in the chosen style (Band, Square, Portrait, One eye, Candid) that never goes below the eyes,
- **fails safe**: frames where the eyes are lost are pixelated and darkened; if the eyes are lost in more than 40% of frames the video is rejected,
- removes sound unless the member turned it on,
- uploads the MP4 + a poster image, then reports back to Supabase.

It holds **no Supabase keys** — the `process-video` edge function gives it short-lived signed URLs, and the
`video-done` edge function receives the result. The only secret is `WORKER_SECRET`, shared with Supabase.

## Deploy on Render (free)

1. Push this `video-worker` folder to a GitHub repo (public, or connect GitHub to Render).
2. New → Web Service → the repo → Runtime **Docker**, plan **Free**, region **Singapore**.
   (If the folder isn't the repo root, set *Root Directory* to `video-worker`.)
3. Environment variable `WORKER_SECRET` = the value in Supabase table `worker_config.secret`.
4. After it deploys, put the service URL (e.g. `https://eyesonly-video.onrender.com`) in `worker_config.url`.

Free instances sleep after 15 minutes idle; the first video after that waits ~1 minute while it wakes.

## Local test

```bash
pip install -r requirements.txt
curl -sSL -o face_detection_yunet_2023mar.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
python -c "import pipeline; print(pipeline.process('in.mp4','out.mp4','poster.jpg','portrait'))"
```
