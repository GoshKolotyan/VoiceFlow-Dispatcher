import pyaudio

import azure.cognitiveservices.speech as speechsdk


# from src.core import logger
from src.config import settings

class AudioProcessor:
    def __init__(self):

        # add here try catch for azure init
        self.speech_config = speechsdk.SpeechConfig(
            subscription=settings.AZURE_SPEECH_KEY,
            region=settings.AZURE_SPEECH_REGION
        )
        self.speech_config.speech_recognition_language = "en-US"
        self.speech_config.speech_synthesis_voice_name = "en-US-DavisNeural"
    
    def listen_from_mic(self):
        "Recording audio from microphone and returning audio data."
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)  
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, 
            audio_config=audio_config
        )
        result = recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # logger.info(f"Recognized: {result.text}")
            return result.text
            
        elif result.reason == speechsdk.ResultReason.NoMatch:
            # logger.warning("No speech could be recognized.")
            return ""
            
        # elif result.reason == speechsdk.ResultReason.Canceled:
        #     cancellation_details = result.cancellation_details
        #     # logger.error(f"Speech Recognition canceled: {cancellation_details.reason}")
        #     if cancellation_details.reason == speechsdk.CancellationReason.Error:
        #         # logger.error(f"Error details: {cancellation_details.error_details}")
        #     return ""
    def speak_text(self, text):
        "Converting text to speech and playing it through speakers."
        audio_config = speechsdk.audio.AudioConfig(use_default_speaker=True)
        
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, 
            audio_config=audio_config
        )

        # logger.info(f"Speaking: {text}")
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            # logger.error(f"Speech Synthesis canceled: {cancellation_details.reason}")

    def __repr__(self):
        return "AudioProcessor()"
    

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    text = audio_processor.listen_from_mic()
    if text:
        audio_processor.speak_text(f"You said: {text}")
    else:
        audio_processor.speak_text("I did not catch that. Please try again.")