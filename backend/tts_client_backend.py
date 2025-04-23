import time
import asyncio
import weakref
import base64
import logging
import io
from pydub import AudioSegment
from typing import AsyncGenerator

from .base_model_backend import BaseModelBackend
from client.tts_client import TTSClient
from concurrent.futures import ThreadPoolExecutor
from services.memory_check import MemoryChecker

class TtsClientBackend(BaseModelBackend):
    SUPPORTED_FORMATS = ["mp3", "opus", "aac", "flac", "wav", "pcm"]

    def __init__(self, model_config):
        super().__init__(model_config)
        self._client_pool = []
        self._active_clients = {}
        self._pool_lock = asyncio.Lock()
        self.logger = logging.getLogger("api.tts")
        self.POOL_SIZE = 1
        self._inference_executor = ThreadPoolExecutor(max_workers=self.POOL_SIZE)
        self._active_tasks = weakref.WeakSet()
        self.memory_checker = MemoryChecker(
            host=self.config["host"],
            port=self.config["port"]
        )
        self.sample_rate = 16000
        self.channels = 1

    async def _get_client(self):
        try:
            await asyncio.wait_for(self._pool_lock.acquire(), timeout=30.0)
            
            start_time = time.time()
            timeout = 30.0
            retry_interval = 3

            while True:
                if self._client_pool:
                    client = self._client_pool.pop()
                    return client
                
                for task in self._active_tasks:
                    task.cancel()
                
                self._pool_lock.release()
                await asyncio.sleep(retry_interval)
                await asyncio.wait_for(self._pool_lock.acquire(), timeout=timeout - (time.time() - start_time))

                client = TTSClient(
                    host=self.config["host"],
                    port=self.config["port"]
                )
                self._active_clients[id(client)] = client

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._inference_executor,
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
                return client
        except asyncio.TimeoutError:
            raise RuntimeError("Server busy, please try again later.")
        finally:
            if self._pool_lock.locked():
                self._pool_lock.release()

    async def _release_client(self, client):
        async with self._pool_lock:
            self._client_pool.append(client)

    async def close(self):
        for task in self._active_tasks:
            task.cancel()
        if self._active_tasks:
            await asyncio.wait(self._active_tasks, timeout=2)
        for client in self._client_pool:
            client.exit()
        self._client_pool.clear()
        self._active_clients.clear()
        self._inference_executor.shutdown(wait=False)

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
        task = asyncio.current_task()
        self._active_tasks.add(task)
        full_data = b''
        try:
            loop = asyncio.get_event_loop()
            async for chunk in client.inference_stream(input_text, object_type="tts.utf-8"):
                pcm_data = base64.b64decode(chunk)
                encoded_data = await loop.run_in_executor(
                    self._inference_executor,
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
            self._active_tasks.discard(task)
            await self._release_client(client)