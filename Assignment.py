# assignment1.py

import argparse
from detector import transcribe_and_detect

def run_cli(audio_file, non_vocal_csv="non-vocal sound.csv"):
    result = transcribe_and_detect(audio_file, non_vocal_csv)
    print(" Transcription complete.")
    print(" Output saved in /results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and detect non-vocal expressions")
    parser.add_argument("audio", nargs="?", help="Path to .wav file")
    args = parser.parse_args()

    if args.audio:
        run_cli(args.audio)
    else:
        print(" Provide a .wav file to run CLI.")

