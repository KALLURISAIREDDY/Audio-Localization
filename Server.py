# server.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from detector import transcribe_and_detect
import uvicorn
import os

app = FastAPI()

@app.post("/transcribe/")
async def transcribe_endpoint(file: UploadFile = File(...)):
    audio_path = f"uploaded_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    non_vocal_csv_path = "non-vocal sound.csv"
    if not os.path.exists(non_vocal_csv_path):
        return JSONResponse(content={"error": "Missing non-vocal sound.csv"}, status_code=400)

    result = transcribe_and_detect(audio_path, non_vocal_csv_path)
    return result

if __name__ == "__main__":
    print(" FastAPI running at http://127.0.0.1:8000/docs")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
