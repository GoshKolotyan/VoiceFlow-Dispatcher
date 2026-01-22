import asyncio
from typing import Callable, Awaitable

from src.core.logger import LoggerFactory
from src.core.expections import SpeechRecognitionError, SpeechSynthesisError
from src.config.settings import Settings
from src.utils.audio import AudioProcessor


class AzureSpeechService:
    """High-level async speech service wrapper for STT/TTS"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = LoggerFactory.create_logger("SpeechService", level=settings.LOG_LEVEL)

        # Use existing AudioProcessor
        self.audio_processor = AudioProcessor()

        # Configuration
        self.default_voice = "en-US-DavisNeural"
        self.max_retries = settings.MAX_RETRIES
        self.timeout = settings.TIMEOUT_SECONDS

        self.logger.info("Initialized Azure Speech service")

    async def speech_to_text(self, timeout: int | None = None) -> str:
        """
        Async wrapper for speech-to-text recognition

        Args:
            timeout: Maximum seconds to wait for speech (uses default if None)

        Returns:
            Recognized text

        Raises:
            SpeechRecognitionError: If recognition fails
        """
        timeout = timeout or self.timeout
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                self.logger.info(f"Listening for speech (attempt {retry_count + 1}/{self.max_retries})...")

                # Run sync method in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    self.audio_processor.listen_from_mic
                )

                if not text or text.strip() == "":
                    if retry_count < self.max_retries - 1:
                        self.logger.warning("No speech recognized, retrying...")
                        retry_count += 1
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        raise SpeechRecognitionError(
                            "No speech could be recognized",
                            details={"attempts": retry_count + 1}
                        )

                self.logger.info(f"Recognized: '{text[:50]}...'")
                return text

            except SpeechRecognitionError:
                # Re-raise our custom exceptions
                raise
            except Exception as e:
                self.logger.error(f"STT failed on attempt {retry_count + 1}: {e}")
                if retry_count < self.max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(1)
                else:
                    raise SpeechRecognitionError(
                        "Speech recognition failed after retries",
                        details={
                            "error": str(e),
                            "attempts": retry_count + 1,
                            "type": type(e).__name__
                        }
                    )

        raise SpeechRecognitionError(
            "Speech recognition failed: max retries exceeded",
            details={"attempts": retry_count}
        )

    async def text_to_speech(
        self,
        text: str,
        voice: str | None = None,
        rate: str = "default"
    ) -> None:
        """
        Async wrapper for text-to-speech synthesis

        Args:
            text: Text to synthesize
            voice: Voice name (uses default if None)
            rate: Speech rate ("slow", "default", "fast")

        Raises:
            SpeechSynthesisError: If synthesis fails
        """
        if not text or text.strip() == "":
            self.logger.warning("Empty text provided for TTS, skipping")
            return

        try:
            # Configure voice if specified
            target_voice = voice or self.default_voice
            if target_voice != self.audio_processor.speech_config.speech_synthesis_voice_name:
                self.audio_processor.speech_config.speech_synthesis_voice_name = target_voice
                self.logger.debug(f"Changed voice to: {target_voice}")

            self.logger.info(f"Synthesizing speech: '{text[:50]}...'")

            # Run sync method in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.audio_processor.speak_text,
                text
            )

            self.logger.info("Speech synthesis completed")

        except Exception as e:
            self.logger.error(f"TTS failed: {e}")
            raise SpeechSynthesisError(
                "Speech synthesis failed",
                details={
                    "text": text[:100],
                    "voice": target_voice,
                    "error": str(e),
                    "type": type(e).__name__
                }
            )

    async def continuous_recognition(
        self,
        callback: Callable[[str], Awaitable[None]],
        duration_seconds: int = 60
    ) -> None:
        """
        Continuous speech recognition with callback (future enhancement)

        Args:
            callback: Async function to call with recognized text
            duration_seconds: How long to listen

        Note: This is a placeholder for future implementation
        """
        self.logger.warning("Continuous recognition not yet implemented")
        # Future: Implement using Azure Speech SDK's continuous recognition
        # with event handlers for recognized phrases
        pass

    async def recognize_keyword(
        self,
        keyword: str,
        timeout: int = 30
    ) -> bool:
        """
        Listen for a specific keyword (future enhancement)

        Args:
            keyword: The keyword to listen for
            timeout: Maximum seconds to wait

        Returns:
            True if keyword was detected, False otherwise

        Note: This is a placeholder for future implementation
        """
        self.logger.warning("Keyword recognition not yet implemented")
        # Future: Implement using Azure Speech SDK's keyword recognition
        return False

    def set_default_voice(self, voice: str) -> None:
        """Change the default voice for TTS"""
        self.default_voice = voice
        self.audio_processor.speech_config.speech_synthesis_voice_name = voice
        self.logger.info(f"Set default voice to: {voice}")

    def get_available_voices(self) -> list[str]:
        """Get list of common available voices"""
        # Common Azure Neural voices for English (US)
        return [
            "en-US-DavisNeural",      # Male, default
            "en-US-JennyNeural",      # Female
            "en-US-GuyNeural",        # Male
            "en-US-AriaNeural",       # Female
            "en-US-TonyNeural",       # Male
            "en-US-SaraNeural",       # Female
        ]

    def __repr__(self) -> str:
        return f"AzureSpeechService(voice={self.default_voice})"
