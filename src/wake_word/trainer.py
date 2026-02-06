"""Wake word model training utility."""
import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class WakeWordTrainer:
    """Records samples and prepares data for wake word training."""

    def __init__(
        self,
        output_dir: str | Path,
        sample_rate: int = 16000,
        duration: float = 2.0,
    ):
        """Initialize trainer.

        Args:
            output_dir: Directory to save recordings
            sample_rate: Audio sample rate
            duration: Recording duration per sample
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._duration = duration
        self._samples_dir = self._output_dir / "samples"
        self._samples_dir.mkdir(exist_ok=True)

    def record_sample(self, name: str, index: int) -> Path:
        """Record a single sample.

        Args:
            name: Wake word name
            index: Sample index

        Returns:
            Path to saved file
        """
        print(f"\n[{index}] Recording in 2 seconds... Say '{name}'")
        time.sleep(2)

        print("Recording... ", end="", flush=True)

        # Record audio
        audio = sd.rec(
            int(self._duration * self._sample_rate),
            samplerate=self._sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()

        print("Done!")

        # Save as WAV
        import wave
        filename = self._samples_dir / f"{name}_{index:03d}.wav"

        audio_int16 = (audio.flatten() * 32767).astype(np.int16)

        with wave.open(str(filename), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return filename

    def record_samples(self, name: str, count: int = 50) -> list[Path]:
        """Record multiple samples interactively.

        Args:
            name: Wake word name
            count: Number of samples to record

        Returns:
            List of saved file paths
        """
        print(f"\nRecording {count} samples for wake word: '{name}'")
        print("Press Ctrl+C to stop early\n")

        samples = []

        try:
            for i in range(count):
                filepath = self.record_sample(name, i)
                samples.append(filepath)
                print(f"Saved: {filepath}")

                if i < count - 1:
                    input("Press Enter for next sample (Ctrl+C to stop)...")

        except KeyboardInterrupt:
            print(f"\n\nStopped. Recorded {len(samples)} samples.")

        return samples

    def record_negative_samples(self, count: int = 20) -> list[Path]:
        """Record negative (non-wake-word) samples.

        Args:
            count: Number of samples

        Returns:
            List of saved file paths
        """
        print(f"\nRecording {count} negative samples (background/other words)")
        print("Say random words or capture background noise\n")

        samples = []

        try:
            for i in range(count):
                filepath = self.record_sample("negative", i)
                samples.append(filepath)
                print(f"Saved: {filepath}")

                if i < count - 1:
                    input("Press Enter for next sample (Ctrl+C to stop)...")

        except KeyboardInterrupt:
            print(f"\n\nStopped. Recorded {len(samples)} negative samples.")

        return samples

    def create_training_manifest(self) -> Path:
        """Create manifest file listing all samples.

        Returns:
            Path to manifest file
        """
        manifest_path = self._output_dir / "manifest.txt"

        positive_samples = list(self._samples_dir.glob("hi_samsung_*.wav"))
        negative_samples = list(self._samples_dir.glob("negative_*.wav"))

        with open(manifest_path, "w") as f:
            f.write("# Positive samples\n")
            for p in sorted(positive_samples):
                f.write(f"1,{p}\n")

            f.write("\n# Negative samples\n")
            for p in sorted(negative_samples):
                f.write(f"0,{p}\n")

        print(f"\nManifest created: {manifest_path}")
        print(f"  Positive samples: {len(positive_samples)}")
        print(f"  Negative samples: {len(negative_samples)}")

        return manifest_path

    def print_training_instructions(self) -> None:
        """Print instructions for training with openWakeWord."""
        print("\n" + "=" * 60)
        print("TRAINING INSTRUCTIONS")
        print("=" * 60)
        print("""
After recording samples, train your custom wake word model:

1. Install openWakeWord training dependencies:
   pip install openwakeword[training]

2. Clone the training repository:
   git clone https://github.com/dscripka/openWakeWord.git
   cd openWakeWord

3. Prepare your data:
   - Copy your samples to the training directory
   - Create positive and negative audio lists

4. Train the model:
   python -m openwakeword.train \\
       --positive_samples /path/to/positive/ \\
       --negative_samples /path/to/negative/ \\
       --output_model hi_samsung.onnx

5. Copy the trained model to:
   models/wake_word/hi_samsung.onnx

For detailed instructions, see:
https://github.com/dscripka/openWakeWord#training-new-models
""")


def main():
    """Run the wake word trainer."""
    setup_logging(level="INFO")

    parser = argparse.ArgumentParser(description="Wake Word Training Utility")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/wake_word/training_data",
        help="Directory for training data",
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        default="hi_samsung",
        help="Wake word name (use underscores)",
    )
    parser.add_argument(
        "--positive-count",
        type=int,
        default=50,
        help="Number of positive samples",
    )
    parser.add_argument(
        "--negative-count",
        type=int,
        default=20,
        help="Number of negative samples",
    )
    parser.add_argument(
        "--record-positive",
        action="store_true",
        help="Record positive samples",
    )
    parser.add_argument(
        "--record-negative",
        action="store_true",
        help="Record negative samples",
    )
    parser.add_argument(
        "--create-manifest",
        action="store_true",
        help="Create training manifest",
    )

    args = parser.parse_args()

    trainer = WakeWordTrainer(
        output_dir=args.output_dir,
    )

    if args.record_positive:
        print(f"\nRecording positive samples for: {args.wake_word}")
        print("Speak clearly and vary your tone/distance")
        trainer.record_samples(args.wake_word, args.positive_count)

    if args.record_negative:
        trainer.record_negative_samples(args.negative_count)

    if args.create_manifest:
        trainer.create_training_manifest()

    if not any([args.record_positive, args.record_negative, args.create_manifest]):
        print("\nNo action specified. Available actions:")
        print("  --record-positive  Record wake word samples")
        print("  --record-negative  Record non-wake-word samples")
        print("  --create-manifest  Create training manifest file")

    trainer.print_training_instructions()


if __name__ == "__main__":
    main()
