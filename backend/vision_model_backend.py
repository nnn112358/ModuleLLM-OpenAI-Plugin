import uuid
import time
from openai import AsyncOpenAI
from .base_model_backend import BaseModelBackend
from .chat_schemas import ChatCompletionRequest, Message, ContentItem
from fastapi import HTTPException
from typing import List

class VisionModelBackend(BaseModelBackend):
    MAX_IMAGE_SIZE = 4 * 1024 * 1024  # 4MB
    IMAGE_TIMEOUT = 15  # 秒
    
    async def _process_image_content(self, content_item: ContentItem) -> dict:
        if not content_item.image_url:
            return None
            
        url = content_item.image_url.get("url", "")
        if url.startswith("data:image"):
            return {
                "type": "image_url",
                "image_url": {"url": url}
            }
            
        # 下载外部图片并转换为base64
        base64_str = await self.download_image(
            url, 
            max_size=self.MAX_IMAGE_SIZE,
            timeout=self.IMAGE_TIMEOUT
        )
        if not base64_str:
            raise HTTPException(
                status_code=400,
                detail=f"无法加载图片: {url}"
            )
            
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_str}"
            }
        }

    async def _build_messages(self, messages: List[Message]):
        processed_messages = []
        
        for msg in messages:
            content = msg.content
            new_content = []
            
            if isinstance(content, list):
                for item in content:
                    if item.type == "text":
                        new_content.append({
                            "type": "text",
                            "text": item.text
                        })
                    elif item.type == "image_url":
                        image_content = await self._process_image_content(item)
                        if image_content:
                            new_content.append(image_content)
            else:
                new_content.append({
                    "type": "text",
                    "text": str(content)
                })
                
            processed_messages.append({
                "role": msg.role,
                "content": new_content
            })
            
        return processed_messages

    async def generate(self, request: ChatCompletionRequest):
        from openai import AsyncOpenAI
        
        try:
            client = AsyncOpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"],
                timeout=30.0
            )
            
            messages = await self._build_messages(request.messages)
            
            response = await client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=request.stream
            )
            
            if request.stream:
                async def stream_wrapper():
                    async for chunk in response:
                        # 统一错误处理
                        if isinstance(chunk, dict) and "error" in chunk:
                            yield chunk
                            continue
                        
                        # 转换为兼容格式
                        yield {
                            "id": f"chatcmpl-{uuid.uuid4()}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": chunk.choices[0].delta.content or "",
                                    "role": "assistant"
                                },
                                "finish_reason": chunk.choices[0].finish_reason
                            }]
                        }
                    yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                return stream_wrapper()
            
            # 非流式响应添加usage信息
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    }
                }],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Vision model error: {str(e)}"
            )