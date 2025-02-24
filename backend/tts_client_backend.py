from concurrent.futures import ThreadPoolExecutor
from .base_model_backend import BaseModelBackend
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

    def _encode_stream_chunk(self, pcm_data: bytes, format: str) -> bytes:
        if format == "pcm":
            return pcm_data
            
        audio = AudioSegment(
            data=pcm_data,
            sample_width=2,
            frame_rate=self.sample_rate,
            channels=self.channels
        )
        buffer = io.BytesIO()
        audio.export(buffer, format=format)
        return buffer.getvalue()

    def _encode_full_audio(self, pcm_data: bytes, format: str) -> bytes:
        audio = AudioSegment(
            data=pcm_data,
            sample_width=2,
            frame_rate=self.sample_rate,
            channels=self.channels
        )
        
        buffer = io.BytesIO()
        audio.export(buffer, format=format)
        return buffer.getvalue()

    def _encode_audio(self, pcm_data: bytes, format: str) -> bytes:
        if format in ["mp3", "opus", "aac", "pcm"]:
            return self._encode_stream_chunk(pcm_data, format)
        
        if not hasattr(self, '_full_audio_buffer'):
            self._full_audio_buffer = io.BytesIO()
        
        self._full_audio_buffer.write(pcm_data)
        
        return b''

    async def generate_speech(self, input_text: str, voice: str = "alloy", format: str = "mp3") -> AsyncGenerator[bytes, None]:
        client = await self._get_client()
        try:
            loop = asyncio.get_event_loop()
            sync_gen = client.inference_stream(input_text, object_type="tts.utf-8")

            def safe_next():
                try:
                    return next(sync_gen)
                except StopIteration:
                    return None

            full_data = b''
            while True:
                chunk = await loop.run_in_executor(self._executor, safe_next)
                if chunk is None:
                    break

                pcm_data = base64.b64decode(chunk)
                encoded_data = await loop.run_in_executor(
                    self._executor,
                    self._encode_audio,
                    pcm_data,
                    format
                )
                if encoded_data:
                    yield encoded_data
                else:
                    full_data += pcm_data

            if format not in ["mp3", "opus", "aac", "pcm"]:
                final_audio = self._encode_full_audio(full_data, format)
                yield final_audio

        finally:
            await self._release_client(client) 