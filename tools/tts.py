import io
import librosa
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import PullAudioOutputStream


class TTSBasedOnAzure(object):
    """ refer to 
        "https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/how-to-speech-synthesis?tabs=browserjs%2Cterminal&pivots=programming-language-python"
    """
    def __init__(self, azure_key: str, azure_region: str, voice_id=None):
        self.azure_key = azure_key
        self.azure_region = azure_region
        self.voice_id = voice_id
        self.sr = 16000

    async def __call__(self, text: str, voice_id: str = None):
        voice_id = self.voice_id if voice_id is None else voice_id
        assert voice_id is not None

        speech_config = speechsdk.SpeechConfig(subscription=self.azure_key, region=self.azure_region)
        speech_config.speech_synthesis_voice_name = voice_id
        audio_stream = PullAudioOutputStream()
        audio_config = speechsdk.audio.AudioOutputConfig(stream=audio_stream)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text)
        try:
            result = result.get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # 关闭流并返回音频数据
                audio_buffer = io.BytesIO(result.audio_data)
                wav = librosa.core.load(audio_buffer, sr=self.sr)[0]
                yield wav
            else:
                if result.reason == speechsdk.ResultReason.Canceled:
                    raise InvalidRequestionParametersError(result.cancellation_details.error_details)
                else:
                    raise InvalidAudioBytesError(f"Invalid audio bytes.")
        finally:
            synthesizer = None


class InvalidRequestionParametersError(Exception):...
class InvalidAudioBytesError(Exception):...
