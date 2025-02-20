from pydantic import BaseModel
from typing import Optional, List, Union
from .chat_schemas import ChatCompletionRequest  # Note: You'll need to move the request models to a schemas.py file

class BaseModelBackend:
    def __init__(self, model_config):
        self.config = model_config

    async def generate(self, request: ChatCompletionRequest):
        raise NotImplementedError