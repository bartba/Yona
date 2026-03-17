#!/usr/bin/env python
"""Live wake word detection test.

Run in a terminal:
    python tests/test_wake_live.py

Say "Hey Jarvis" into the Poly Sync 20 microphone.
Detected events are printed in real time. Press Ctrl+C to stop.
"""

import sys
import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

# --- Configuration ---
MODEL_PATH = "models/wake_word/hey_jarvis_v0.1.onnx"
DEVICE_NAME = "Poly Sync 20"
SAMPLE_RATE = 16000
CHUNK_SIZE = 512       # 32 ms @ 16 kHz
THRESHOLD = 0.5
PATIENCE = 3
DURATION = 60          # seconds (Ctrl+C to stop early)


def find_device(name: str) -> int:
    """Find audio input device index by name."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if name in dev["name"] and dev["max_input_channels"] > 0:
            return i
    print(f"ERROR: '{name}' not found. Available input devices:")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']}")
    sys.exit(1)


def main() -> None:
    device_idx = find_device(DEVICE_NAME)
    print(f"Audio device: [{device_idx}] {DEVICE_NAME}")

    model = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
    model_names = list(model.models.keys())
    print(f"Loaded models: {model_names}")
    print(f"Threshold: {THRESHOLD}, Patience: {PATIENCE}")
    print(f"Listening for {DURATION}s... Say 'Hey Jarvis'! (Ctrl+C to stop)\n")

    detected_count = 0

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal detected_count
        if status:
            print(f"  [audio status: {status}]", flush=True)

        audio = indata[:, 0].astype(np.float32)
        pcm = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)

        result = model.predict(
            pcm,
            threshold={name: THRESHOLD for name in model_names},
            patience={name: PATIENCE for name in model_names},
        )

        for name in model_names:
            score = result.get(name, 0.0)
            if score >= THRESHOLD:
                detected_count += 1
                print(
                    f"  *** DETECTED '{name}' score={score:.3f} "
                    f"(#{detected_count}) ***",
                    flush=True,
                )
            elif score > 0.05:
                print(f"  {name} score={score:.3f}", flush=True)

    try:
        with sd.InputStream(
            device=device_idx,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=callback,
        ):
            start = time.time()
            while time.time() - start < DURATION:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    print(f"\nDone. Detected {detected_count} time(s).")


if __name__ == "__main__":
    main()
