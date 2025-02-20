from pydantic import BaseModel
from typing import Optional, List, Union

class ContentItem(BaseModel):
    type: str  # text/image_url
    text: Optional[str] = None
    image_url: Optional[dict] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False 