from typing import Any, List, Optional, Type


class DocumentSourceContent:
    content_type: str
    media_type: str
    data: str

    def __init__(self, content_type: str, media_type: str, data: str) -> None: ...

    def __repr__(self) -> str: ...


class DocumentContent:
    content_type: str
    source: DocumentSourceContent

    def __init__(self, content_type: str, source: DocumentSourceContent) -> None: ...

    def __repr__(self) -> str: ...


class TextContent:
    content_type: str
    text: str

    def __init__(self, text: str) -> None: ...

    def __repr__(self) -> str: ...


class Content:
    @classmethod
    def from_text(cls: Type['Content'], text: str) -> 'Content': ...

    @classmethod
    def from_document(cls: Type['Content'], *args: Any, **kwargs: Any) -> 'Content': ...

    def __repr__(self) -> str: ...


class Message:
    role: str
    content: List[Content]

    def __init__(self, role: str, content: List[Content]) -> None: ...

    def __repr__(self) -> str: ...


class AnthropicRequest:
    model: str
    max_tokens: int
    messages: List[Message]

    def __init__(self, model: str, max_tokens: int, messages: List[Message]) -> None: ...

    def __repr__(self) -> str: ...


class ResponseContent:
    content_type: str
    text: str

    def __repr__(self) -> str: ...


class Usage:
    input_tokens: int
    output_tokens: int

    def __repr__(self) -> str: ...


class AnthropicResponse:
    id: str
    response_type: str
    role: str
    content: List[ResponseContent]
    model: str
    stop_reason: Optional[str]
    usage: Usage

    def __repr__(self) -> str: ...


def get_response(request_body: AnthropicRequest) -> AnthropicResponse: ...
