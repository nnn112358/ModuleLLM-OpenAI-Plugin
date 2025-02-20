import uuid
import time
from openai import AsyncOpenAI
from .base_model_backend import BaseModelBackend
from .chat_schemas import ChatCompletionRequest
from fastapi import HTTPException

class OpenAIProxyBackend(BaseModelBackend):
    async def generate(self, request: ChatCompletionRequest):
        from openai import AsyncOpenAI, APIError
        
        try:
            client = AsyncOpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"]
            )
            
            response = await client.chat.completions.create(
                model=self.config["model"],
                messages=[m.model_dump() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream
            )
            
            if request.stream:
                async def async_wrapper():
                    try:
                        async for chunk in response:
                            yield chunk
                    except APIError as e:
                        yield {
                            "error": {
                                "message": f"OpenAI API Error: {str(e)}",
                                "type": "api_error"
                            }
                        }
                return async_wrapper()
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI proxy error: {str(e)}"
            )