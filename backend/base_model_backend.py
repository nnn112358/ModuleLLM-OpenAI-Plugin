from pydantic import BaseModel
from typing import Optional, List, Union
from .chat_schemas import ChatCompletionRequest

class BaseModelBackend:
    def __init__(self, model_config):
        self.config = model_config

    async def generate(self, request: ChatCompletionRequest):
        raise NotImplementedError