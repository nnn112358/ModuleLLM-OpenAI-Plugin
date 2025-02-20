import uuid
import time
from .base_model_backend import BaseModelBackend
from .chat_schemas import ChatCompletionRequest

class TestBackend(BaseModelBackend):
    async def generate(self, request: ChatCompletionRequest):
        if request.stream:
            async def chunk_generator():
                content_parts = ["🤣", "👉🏻", "🤡"]
                messages=[m.model_dump() for m in request.messages]
                print(f"messages:_____________{messages}______________")
                for i, part in enumerate(content_parts):
                    yield {
                        "id": f"chatcmpl-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": part,
                                "role": "assistant" if i == 0 else None,
                                "function_call": None,
                                "tool_calls": None
                            },
                            "logprobs": None,
                            "finish_reason": "stop" if i == len(content_parts)-1 else None
                        }],
                        "service_tier": None,
                        "system_fingerprint": None,
                        "usage": None
                    }
            return chunk_generator()
        else:
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "🤣👉🏻🤡",
                        "function_call": None,
                        "tool_calls": None
                    },
                    "finish_reason": "stop",
                    "index": 0
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }