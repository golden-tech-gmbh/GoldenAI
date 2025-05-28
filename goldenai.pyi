from typing import Any, List, Type, Dict


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
    def from_text(cls: Type["Content"], text: str) -> "Content": ...

    @classmethod
    def from_document(cls: Type["Content"], *args: Any, **kwargs: Any) -> "Content": ...

    def __repr__(self) -> str: ...


class Message:
    role: str
    content: List[Content]

    def __init__(self, content: List[Content]) -> None: ...

    def __repr__(self) -> str: ...


class AnthropicRequest:
    model: str
    max_tokens: int
    messages: List[Message]

    def __init__(
            self,
            model: str,
            max_tokens: int,
            messages: List[Message],
            prompt: str | None = None,
    ) -> None: ...

    def __repr__(self) -> str: ...


class OpenAIRequest:
    model: str
    max_tokens: int
    messages: List[Message]

    def __init__(
            self,
            model: str,
            max_tokens: int,
            messages: List[Message],
            prompt: str | None = None,
    ) -> None: ...

    def __repr__(self) -> str: ...


class ResponseContent:
    content_type: str
    text: str

    def __repr__(self) -> str: ...


class ResponseMsgOpenAI:
    role: str
    content: str

    def __repr__(self) -> str: ...


class ResponseChoiceOpenAI:
    index: int
    message: ResponseMsgOpenAI
    finish_reason: str | None

    def __repr__(self) -> str: ...


class LLMResponse:
    id: str
    model: str
    response_type: str
    role: str | None
    content: List[ResponseContent] | None
    choices: List[ResponseChoiceOpenAI] | None
    stop_reason: str | None
    usage: Dict[str, int]

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...


class LLM:
    def __init__(self) -> None: ...

    Anthropic: LLM
    OpenAI: LLM


def send(model: LLM, request_body: AnthropicRequest | OpenAIRequest) -> LLMResponse:
    """
    Send prepared LLM Request
    :param model: 
    :param request_body: AnthropicRequest or OpenAIRequest
    :return: LLMResponse
    """
