from concurrent.futures import ThreadPoolExecutor
from .base_model_backend import BaseModelBackend
from .chat_schemas import ChatCompletionRequest
from client.tts_client import TTSClient
import asyncio
import base64
import logging
import io
from pydub import AudioSegment
import numpy as np
from typing import AsyncGenerator

class TtsClientBackend(BaseModelBackend):
    POOL_SIZE = 1
    SUPPORTED_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

    def __init__(self, model_config):
        super().__init__(model_config)
        self._client_pool = []
        self._active_clients = {}
        self._pool_lock = asyncio.Lock()
        self.logger = logging.getLogger("api.tts")
        self._executor = ThreadPoolExecutor(max_workers=self.POOL_SIZE)
        self.sample_rate = 16000
        self.channels = 1

    async def _get_client(self):
        async with self._pool_lock:
            if self._client_pool:
                return self._client_pool.pop()
            
            if len(self._active_clients) >= self.POOL_SIZE:
                raise RuntimeError("TTS connection pool exhausted")

            client = TTSClient(
                host=self.config["host"],
                port=self.config["port"]
            )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: client.setup(
                    "melotts.setup",
                    {
                        "model": self.config["model_name"],
                        "response_format": "pcm.stream.base64",
                        "input": "tts.utf-8",
                        "enoutput": True,
                        "voice": "alloy"
                    }
                )
            )
            
            self._active_clients[id(client)] = client
            return client

    async def _release_client(self, client):
        async with self._pool_lock:
            self._client_pool.append(client)

    def _encode_audio(self, pcm_data: bytes, format: str) -> bytes:
        raw_audio = np.frombuffer(pcm_data, dtype=np.int16)
        audio_segment = AudioSegment(
            raw_audio.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=self.channels
        )

        format_map = {
            "mp3": "mp3",
            "opus": "ogg",
            "aac": "adts",
            "flac": "flac",
            "wav": "wav",
            "pcm": "raw"
        }

        if format == "pcm":
            return pcm_data

        buffer = io.BytesIO()
        audio_segment.export(buffer, format=format_map[format])
        return buffer.getvalue()

    async def generate_speech(self, input_text: str, voice: str = "alloy", format: str = "mp3") -> AsyncGenerator[bytes, None]:
        client = await self._get_client()
        try:
            loop = asyncio.get_event_loop()
            sync_gen = client.inference_stream(input_text, object_type="tts.utf-8")

            while True:
                chunk = await loop.run_in_executor(self._executor, next, sync_gen)
                pcm_data = base64.b64decode(chunk)

                encoded_data = await loop.run_in_executor(
                    self._executor,
                    self._encode_audio,
                    pcm_data,
                    format
                )

                yield encoded_data
                break

        finally:
            await self._release_client(client) 