import os
import uuid
import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
import time
import json
import asyncio
from backend.test_backend import TestBackend
from backend.openai_proxy_backend import OpenAIProxyBackend 
from backend.llm_client_backend import LlmClientBackend
from backend.vision_model_backend import VisionModelBackend
from backend.chat_schemas import ChatCompletionRequest, CompletionRequest, Message, ContentItem

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("api")

app = FastAPI(title="OpenAI Compatible API Server")
limiter = Limiter(key_func=get_remote_address)

class Config:
    def __init__(self):
        with open("config/config.yaml") as f:
            self.data = yaml.safe_load(f)

config = Config()

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/v1"):
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        if api_key != os.getenv("API_KEY"):
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid authentication credentials"}
            )
    return await call_next(request)

class ModelDispatcher:
    def __init__(self):
        self.backends = {}
        self.load_models()

    def load_models(self):
        for model_name, model_config in config.data["models"].items():
            if model_config["type"] == "openai_proxy":
                self.backends[model_name] = OpenAIProxyBackend(model_config)
            elif model_config["type"] == "tcp_client":
                self.backends[model_name] = LlmClientBackend(model_config)
            elif model_config["type"] == "llama.cpp":
                self.backends[model_name] = TestBackend(model_config)
            elif model_config["type"] == "vision_model":
                self.backends[model_name] = VisionModelBackend(model_config)

    def get_backend(self, model_name):
        return self.backends.get(model_name)

_dispatcher = ModelDispatcher()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    backend = _dispatcher.get_backend(body.model)
    if not backend:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported model: {body.model}"
        )
    
    try:
        print(f"Received request: {body.model_dump()}")
        
        if body.stream:
            chunk_generator = await backend.generate(body)
            if not chunk_generator:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate stream response"
                )
            
            async def format_stream():
                try:
                    async for chunk in chunk_generator:
                        if isinstance(chunk, dict):
                            chunk_dict = chunk
                        else:
                            chunk_dict = chunk.model_dump()
                            
                        json_chunk = json.dumps(chunk_dict, ensure_ascii=False)
                        print(f"Sending chunk: {json_chunk}")
                        yield f"data: {json_chunk}\n\n"
                except asyncio.CancelledError:
                    logger.warning("Client disconnected early, terminating inference...")
                    if backend and isinstance(backend, LlmClientBackend):
                        for task in backend._active_tasks:
                            task.cancel()
                    raise
                finally:
                    logger.debug("Stream connection closed")

            return StreamingResponse(
                format_stream(),
                media_type="text/event-stream"
            )
        else:
            response = await backend.generate(body)
            print(f"Sending response: {response}")
            return JSONResponse(content=response)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def create_completion(request: Request, body: CompletionRequest):
    chat_request = ChatCompletionRequest(
        model=body.model,
        messages=[Message(role="user", content=body.prompt)],
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_p=body.top_p,
        stream=body.stream
    )
    
    backend = _dispatcher.get_backend(chat_request.model)
    if not backend:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {chat_request.model}")

    try:
        if body.stream:
            chunk_generator = await backend.generate(chat_request)
            
            async def convert_stream():
                async for chunk in chunk_generator:
                    # Convert format and serialize to JSON string
                    completion_chunk = {
                        "id": chunk.get("id", f"cmpl-{uuid.uuid4()}"),
                        "object": "text_completion.chunk",
                        "created": chunk.get("created", int(time.time())),
                        "model": chat_request.model,
                        "choices": [{
                            "text": chunk["choices"][0]["delta"].get("content", ""),
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": chunk["choices"][0].get("finish_reason")
                        }]
                    }
                    yield f"data: {json.dumps(completion_chunk)}\n\n"
                
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                convert_stream(),
                media_type="text/event-stream"
            )
        else:
            chat_response = await backend.generate(chat_request)
            return JSONResponse({
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": chat_request.model,
                "choices": [{
                    "text": chat_response["choices"][0]["message"]["content"],
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": chat_response.get("usage", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                })
            })
            
    except Exception as e:
        logger.error(f"Completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

logging.getLogger().handlers[0].flush()