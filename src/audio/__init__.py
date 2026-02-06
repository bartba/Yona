"""Audio handling modules."""
from .audio_buffer import AudioBuffer
from .vad import VoiceActivityDetector

# Lazy import for AudioManager (requires sounddevice/PortAudio)
def __getattr__(name):
    if name == "AudioManager":
        from .audio_manager import AudioManager
        return AudioManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AudioManager", "AudioBuffer", "VoiceActivityDetector"]
