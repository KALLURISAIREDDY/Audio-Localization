from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from threading import Thread
import uvicorn
import os
import uuid

from detector import transcribe_and_detect

app = FastAPI()

jobs = {}  # job_id → status & results

def process_job(job_id, audio_path, non_vocal_csv_path):
    try:
        jobs[job_id] = {"status": "processing"}
        result = transcribe_and_detect(audio_path, non_vocal_csv_path)
        jobs[job_id] = {"status": "completed", "result": result}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.post("/transcribe/")
async def transcribe_endpoint(file: UploadFile = File(...)):
    non_vocal_csv_path = "non-vocal sound.csv"
    if not os.path.exists(non_vocal_csv_path):
        return JSONResponse(content={"error": "Missing non-vocal sound.csv"}, status_code=400)

    job_id = str(uuid.uuid4())
    audio_path = f"uploaded_{job_id}_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    thread = Thread(target=process_job, args=(job_id, audio_path, non_vocal_csv_path))
    thread.start()

    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        return JSONResponse(content={"error": "Invalid job_id"}, status_code=404)
    return jobs[job_id]

if __name__ == "__main__":
    print("🚀 FastAPI running at http://0.0.0.0:8000/docs")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
