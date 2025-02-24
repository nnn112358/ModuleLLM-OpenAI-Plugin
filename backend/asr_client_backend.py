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
        loop = asyncio.get_event_loop()
        
        client = await self._get_client()
        
        try:
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            await loop.run_in_executor(
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
            
            full_text = ""
            for chunk in client.inference_stream(audio_b64, object_type="asr.base64"):
                full_text += chunk
            
            return full_text
            
        finally:
            await self._release_client(client)
    
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
            return client
            
    async def _release_client(self, client):
        async with self._lock:
            self.clients.append(client)