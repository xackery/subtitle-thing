#!/usr/bin/env python3
import argparse
import os
from datetime import timedelta

from faster_whisper import WhisperModel


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    # SRT format: HH:MM:SS,mmm
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg.start)
            end = format_timestamp(seg.end)
            text = seg.text.strip()

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(text + "\n\n")


def write_txt(segments, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{format_timestamp(seg.start)}] {seg.text.strip()}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video to SRT using SYSTRAN/faster-whisper"
    )
    parser.add_argument("input", help="Input audio/video file (e.g. .mp4, .mkv, .wav)")
    parser.add_argument(
        "--model-size",
        default="large-v3",
        help="Model size (e.g. tiny, base, small, medium, large-v2, large-v3, distil-large-v3)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="Compute type (e.g. float16, int8_float16, int8, float32)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. en). If omitted, model will auto-detect.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs (default: same as input file)",
    )

    args = parser.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        raise SystemExit(f"Input not found: {in_path}")

    base_dir = args.output_dir or os.path.dirname(os.path.abspath(in_path)) or "."
    base_name = os.path.splitext(os.path.basename(in_path))[0]

    srt_path = os.path.join(base_dir, base_name + ".srt")
    txt_path = os.path.join(base_dir, base_name + ".txt")

    print(f"Loading model '{args.model_size}' on {args.device} ({args.compute_type})...")
    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    print(f"Transcribing: {in_path}")
    segments, info = model.transcribe(
        in_path,
        beam_size=args.beam_size,
        language=args.language,
        vad_filter=True,  # skip long silences
    )

    segments = list(segments)  # force evaluation so we can write outputs
    print(f"Detected language: {info.language} (p={info.language_probability:.3f})")
    print(f"Got {len(segments)} segments; writing SRT/TXT...")

    os.makedirs(base_dir, exist_ok=True)
    write_srt(segments, srt_path)
    write_txt(segments, txt_path)

    print(f"Wrote:\n  {srt_path}\n  {txt_path}")


if __name__ == "__main__":
    main()
