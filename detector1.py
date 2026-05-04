import os
import pandas as pd
import json
import re
import librosa
import soundfile as sf
import nemo.collections.asr as nemo_asr

asr_model = None

def normalize(text):
    text = text.lower().strip()
    return re.sub(r'[^a-z0-9]+', '', text)

def convert_to_mono(input_path, output_path):
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    sf.write(output_path, audio, sr)
    return output_path

def transcribe_and_detect(audio_path, non_vocal_csv_path="non-vocal sound.csv"):
    global asr_model

    if asr_model is None:
        print("🔁 Loading ASR model (nvidia/parakeet-tdt-0.6b-v2)...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

    os.makedirs("results", exist_ok=True)
    mono_path = os.path.splitext(audio_path)[0] + "_mono.wav"
    convert_to_mono(audio_path, mono_path)

    print("🎧 Transcribing...")
    result = asr_model.transcribe([mono_path], timestamps=True)
    segments = result[0].timestamp['segment']
    words = result[0].timestamp['word']

    non_vocal_df = pd.read_csv(non_vocal_csv_path)
    non_vocal_sounds = set(normalize(str(s)) for s in non_vocal_df['sound'] if isinstance(s, str))

    buffer = 0.1
    all_sentences = []
    detected_only = []

    for seg in segments:
        seg_start = float(seg['start'])
        seg_end = float(seg['end'])
        sentence = seg['segment']
        matched = []

        for word in words:
            word_start = float(word['start'])
            word_end = float(word['end'])
            word_text = word['word']
            word_clean = normalize(word_text)

            if seg_start - buffer <= word_start <= seg_end + buffer:
                if word_clean in non_vocal_sounds:
                    match = {
                        "text": word_text,
                        "start_time": f"{word_start:.3f}",
                        "end_time": f"{word_end:.3f}"
                    }
                    matched.append(match)
                    detected_only.append(match)

        all_sentences.append({
            "start_time": f"{seg_start:.3f}",
            "end_time": f"{seg_end:.3f}",
            "sentence": sentence,
            "non_vocal_sounds": matched
        })

    csv_all = "results/all_sentences_with_non_vocal_word_timestamp.csv"
    csv_detected = "results/non_vocal_detected_words.csv"
    json_detected = "results/non_vocal_detected_words.json"

    pd.DataFrame(all_sentences).to_csv(csv_all, index=False)
    pd.DataFrame(detected_only).to_csv(csv_detected, index=False)
    with open(json_detected, "w") as f:
        json.dump(detected_only, f, indent=4)

    return {
        "summary": f"Processed {audio_path}",
        "non_vocal_count": len(detected_only),
        "csv_all_sentences": csv_all,
        "csv_detected": csv_detected,
        "json_detected": json_detected
    }
