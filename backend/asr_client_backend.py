from .base_model_backend import BaseModelBackend
from client.asr_client import LLMClient
import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("api.asr")

class ASRClientBackend(BaseModelBackend):
    POOL_SIZE = 1
    SUPPORTED_FORMATS = ["json", "text", "srt", "verbose_json"]
    
    def __init__(self, model_config):
        super().__init__(model_config)
        self._executor = ThreadPoolExecutor(max_workers=self.POOL_SIZE)
        self.clients = []
        self._lock = asyncio.Lock()
        
    async def create_transcription(self, audio_data: bytes, language: str = "zh", prompt: str = "") -> str:
        client = await self._get_client()
        try:
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            return await self._inference_stream(client, audio_b64)
        finally:
            await self._release_client(client)

    async def _inference_stream(self, client, audio_b64: str) -> str:
        loop = asyncio.get_event_loop()
        full_text = ""
        for chunk in await loop.run_in_executor(
            self._executor,
            client.inference_stream,
            audio_b64,
            "asr.base64"
        ):
            full_text += chunk
        return full_text

    async def _get_client(self):
        async with self._lock:
            if self.clients:
                return self.clients.pop()
            
            if len(self.clients) >= self.POOL_SIZE:
                raise RuntimeError("ASR connection pool exhausted")
                
            client = LLMClient(
                host=self.config["host"],
                port=self.config["port"]
            )
            
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                client.setup,
                "whisper.setup",
                {
                    "model": self.config["model_name"],
                    "response_format": "asr.utf-8",
                    "input": "whisper.base64",
                    "language": "zh",
                    "enoutput": True
                }
            )
            return client

    async def _release_client(self, client):
        async with self._lock:
            self.clients.append(client)