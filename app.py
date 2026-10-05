"""
EyesOnly video worker (Render web service).

POST /process  (header X-Worker-Secret)
  { post_id, style, sound, raw_url, upload_video_url, upload_poster_url, callback_url }
  -> 202 immediately; the job runs in the background, then POSTs the result to callback_url.

The worker holds no Supabase keys: the edge function hands it short-lived signed URLs.
"""
import hmac
import os
import tempfile
import threading
import traceback

import requests
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import pipeline

SECRET = os.environ.get("WORKER_SECRET", "")
MAX_DOWNLOAD = 80 * 1024 * 1024  # 80 MB
app = FastAPI()
_jobs = threading.Semaphore(1)  # one video at a time on a small instance


class Job(BaseModel):
    post_id: str
    style: str = "band"
    sound: bool = False
    raw_url: str
    upload_video_url: str
    upload_poster_url: str
    callback_url: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process", status_code=202)
def process(job: Job, x_worker_secret: str = Header(default="")):
    if not SECRET or not hmac.compare_digest(x_worker_secret, SECRET):
        raise HTTPException(status_code=401)
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return {"accepted": True}


def _callback(job: Job, payload: dict):
    try:
        requests.post(job.callback_url, json={"post_id": job.post_id, **payload},
                      headers={"X-Worker-Secret": SECRET}, timeout=30)
    except Exception:
        traceback.print_exc()


def _run(job: Job):
    with _jobs, tempfile.TemporaryDirectory() as tmp:
        src, out, poster = (os.path.join(tmp, n) for n in ("in.mp4", "out.mp4", "poster.jpg"))
        try:
            with requests.get(job.raw_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                size = 0
                with open(src, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        size += len(chunk)
                        if size > MAX_DOWNLOAD:
                            raise pipeline.VideoRejected("That video file is too large.")
                        f.write(chunk)
            info = pipeline.process(src, out, poster, job.style, job.sound)
            for url, path, ctype in ((job.upload_video_url, out, "video/mp4"), (job.upload_poster_url, poster, "image/jpeg")):
                with open(path, "rb") as f:
                    resp = requests.put(url, data=f, headers={"Content-Type": ctype, "x-upsert": "true"}, timeout=120)
                resp.raise_for_status()
            _callback(job, {"ok": True, **info})
        except pipeline.VideoRejected as e:
            _callback(job, {"ok": False, "error": str(e)})
        except Exception:
            traceback.print_exc()
            _callback(job, {"ok": False, "error": "Something went wrong processing your video. Please try again."})
