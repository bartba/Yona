"""Text-to-speech using Edge TTS."""
import asyncio
import io
import tempfile
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Synthesizer:
    """Text-to-speech synthesis using Edge TTS."""

    def __init__(
        self,
        voices: dict[str, str] | None = None,
        default_voice: str = "ko-KR-SunHiNeural",
        match_input_language: bool = True,
    ):
        """Initialize synthesizer.

        Args:
            voices: Mapping of language code to voice name
            default_voice: Default voice to use
            match_input_language: Whether to match TTS voice to input language
        """
        self._voices = voices or {
            "ko": "ko-KR-SunHiNeural",
            "en": "en-US-JennyNeural",
        }
        self._default_voice = default_voice
        self._match_input_language = match_input_language

    def _get_voice(self, language: str) -> str:
        """Get voice for language.

        Args:
            language: Language code (ko, en, etc.)

        Returns:
            Voice name
        """
        if not self._match_input_language:
            return self._default_voice

        # Normalize language code
        lang = language.lower()[:2]

        return self._voices.get(lang, self._default_voice)

    async def synthesize(
        self,
        text: str,
        language: str = "ko",
    ) -> tuple[np.ndarray, int]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize
            language: Language of the text

        Returns:
            Tuple of (audio samples, sample rate)
        """
        import edge_tts
        from pydub import AudioSegment

        voice = self._get_voice(language)
        logger.info(f"Synthesizing with voice: {voice}")

        # Create TTS communicate
        communicate = edge_tts.Communicate(text, voice)

        # Collect audio chunks
        audio_data = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])

        audio_data.seek(0)

        # Convert MP3 to numpy array
        try:
            audio_segment = AudioSegment.from_mp3(audio_data)

            # Convert to numpy
            samples = np.array(audio_segment.get_array_of_samples())

            # Normalize to float32
            if audio_segment.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:
                samples = samples.astype(np.float32) / 2147483648.0
            else:
                samples = samples.astype(np.float32) / 255.0

            # Handle stereo
            if audio_segment.channels == 2:
                samples = samples.reshape(-1, 2)

            sample_rate = audio_segment.frame_rate

            logger.debug(
                f"Synthesized {len(samples)/sample_rate:.2f}s "
                f"at {sample_rate}Hz"
            )

            return samples, sample_rate

        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        language: str = "ko",
    ) -> Path:
        """Synthesize text to audio file.

        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            language: Language of the text

        Returns:
            Path to the saved file
        """
        import edge_tts

        voice = self._get_voice(language)
        output_path = Path(output_path)

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))

        logger.info(f"Saved TTS audio to: {output_path}")
        return output_path

    @staticmethod
    async def list_voices(language_filter: str | None = None) -> list[dict]:
        """List available voices.

        Args:
            language_filter: Filter by language code (e.g., 'ko', 'en')

        Returns:
            List of voice info dicts
        """
        import edge_tts

        voices = await edge_tts.list_voices()

        if language_filter:
            voices = [
                v for v in voices
                if v["Locale"].lower().startswith(language_filter.lower())
            ]

        return voices


async def test_synthesizer():
    """Test TTS synthesis."""
    synth = Synthesizer()

    # List Korean voices
    print("Korean voices:")
    voices = await Synthesizer.list_voices("ko")
    for v in voices[:5]:
        print(f"  {v['ShortName']}: {v['Gender']}")

    # Test synthesis
    print("\nSynthesizing Korean text...")
    audio, sr = await synth.synthesize("안녕하세요, 저는 요나입니다.", "ko")
    print(f"Generated {len(audio)/sr:.2f}s at {sr}Hz")

    print("\nSynthesizing English text...")
    audio, sr = await synth.synthesize("Hello, I am Yona.", "en")
    print(f"Generated {len(audio)/sr:.2f}s at {sr}Hz")


if __name__ == "__main__":
    asyncio.run(test_synthesizer())
