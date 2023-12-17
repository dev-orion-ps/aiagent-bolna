import os
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError
from aiobotocore.session import AioSession
from contextlib import AsyncExitStack
from agents.helpers.logger_config import configure_logger
from .base_synthesizer import BaseSynthesizer


logger = configure_logger(__name__)
load_dotenv()


class PollySynthesizer(BaseSynthesizer):
    def __init__(self, model, audio_format, voice, language, sampling_rate, stream=False, buffer_size=400):
        super().__init__(stream, buffer_size)
        self.model = model
        self.format = audio_format
        self.voice = voice
        self.language = '{}-IN'.format(language)
        self.sample_rate = sampling_rate

        # @TODO: initialize client here
        self.client = None

    # @TODO: remove AWS client passed as params
    @staticmethod
    async def create_client(service: str, session: AioSession, exit_stack: AsyncExitStack):
        if os.getenv('AWS_ACCESS_KEY_ID'):
            return await exit_stack.enter_async_context(session.create_client(
                service,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION')
            ))
        else:
            return await exit_stack.enter_async_context(session.create_client(service))

    async def generate_tts_response(self, text):
        session = AioSession()

        async with AsyncExitStack() as exit_stack:
            polly = await self.create_client("polly", session, exit_stack)
            try:
                response = await polly.synthesize_speech(
                    Engine='neural',
                    Text=text,
                    OutputFormat=self.format,
                    VoiceId=self.voice,
                    LanguageCode=self.language,
                    SampleRate=self.sample_rate
                )
            except (BotoCoreError, ClientError) as error:
                logger.error(error)
            else:
                return await response["AudioStream"].read()

    async def generate(self, text):
        while True:
            try:
                if text != "" and text != "LLM_END":
                    chunk = await self.generate_tts_response(text)
                    return chunk
            except Exception as e:
                logger.debug("Exception occurred in generate polly: {}".format(e))
                continue