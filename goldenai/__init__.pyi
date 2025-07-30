from typing import List, Type, Dict
from pydantic import BaseModel

from .py.struct import GoldenAIParsedResponse


class DocumentSourceContent:
    content_type: str
    media_type: str
    data: str

    def __repr__(self) -> str: ...


class DocumentContent:
    content_type: str
    source: DocumentSourceContent

    def __init__(self, path: str) -> None:
        """
        Initialize a DocumentContent object.

        Args:
            path (str): The path to the document to create the DocumentContent object from.
        """
        ...

    def __repr__(self) -> str: ...


class TextContent:
    content_type: str
    text: str

    def __init__(self, text: str) -> None:
        """
        Create a TextContent object.

        :param text: The text to create the TextContent object from.
        """
        ...

    def __repr__(self) -> str: ...


class Content:

    def __init__(self, object: TextContent | DocumentContent):
        """
        Initialize a Content object.
        """
        ...

    @classmethod
    def from_text(cls: Type["Content"], text: str) -> "Content":
        """
        Create a Content object from a string.

        Args:
            text (str): The string to create the Content object from.

        Returns:
            Content: The Content object created from the string.
        """
        ...

    @classmethod
    def from_document(cls: Type["Content"], path: str, llm: str | None = None) -> "Content":
        """
        Create a Content object from a document.

        Args:
            path (str): The path to the document to create the Content object from.
            llm (SupportedModels str | None, optional): The language model to use, defaults to None, which will use the GPT41Nano

        Returns:
            Content: The Content object created from the document.
        """
        ...

    def __repr__(self) -> str: ...


class Message:
    role: str
    content: List[Content]

    def __init__(self, content: List[Content]) -> None:
        """
        Initialize a Message object.

        Args:
            content (List[Content]): The content of the message.
        """
        ...

    def __repr__(self) -> str: ...


class AnthropicRequest:
    model: str
    max_tokens: int
    messages: List[Message]
    prompt: str | None

    def __init__(
            self,
            model: str,
            messages: List[Message],
            max_tokens: int | None = 1024,
            prompt: str | None = None,
    ) -> None:
        """
        Initialize an AnthropicRequest object.

        Args:
            model (str): The name of the AI model to use.
            messages (List[Message]): The conversation history.
            max_tokens (int | None, optional): The maximum number of tokens to generate.
            prompt (str | None, optional): The initial prompt for the AI model.
                Defaults to None.
        """
        ...

    def __repr__(self) -> str: ...

    def add_message(self, message: Message) -> None:
        """
        Append a message to the response that will be sent to the LLM in the chat mode.
        :param message: The message to add.
        """

    def add_response(self, response: LLMResponse) -> None:
        """
        Append a response to the response that will be sent to the LLM in the chat mode.
        :param response:
        """


class OpenAIRequest:
    model: str
    messages: List[Message]
    prompt: str | None
    endpoint: str | None

    def __init__(
            self,
            model: str,
            messages: List[Message],
            prompt: str | None = None,
            endpoint: str | None = None
    ) -> None:
        """
        Initialize an OpenAIRequest object.

        Args:
            model (str): The name of the AI model to use.
            messages (List[Message]): The conversation history.
            prompt (str | None, optional): The initial prompt for the AI model.
                Defaults to None.
            endpoint (str | None, optional): The endpoint to use for the OpenAI API.
                Defaults to None, which uses the default OpenAI endpoint.
        """
        ...

    def __repr__(self) -> str: ...

    @property
    def model(self) -> str:
        """
        returns the model name
        :return: model name
        """

    def add_message(self, message: Message) -> None:
        """
        Append a message to the response that will be sent to the LLM in the chat mode.
        :param message: The message to add.
        """

    def add_response(self, response: LLMResponse) -> None:
        """
        Append a response to the response that will be sent to the LLM in the chat mode.
        :param response:
        """

    def add_response_from_str(self, response: str) -> None:
        """
        Add a response string to the response that will be sent to the LLM in the chat mode.
        :param response: str, the response string, can be obtained by str(response)
        """


class OllamaRequest:
    url: str
    model: str
    messages: List[Message]
    prompt: str | None

    def __init__(
            self,
            url: str,
            model: str,
            messages: List[Message],
            prompt: str | None = None,
            image: str | None = None
    ) -> None:
        """
        Initialize an OllamaRequest object.

        Args:
            url (str): The URL of the Ollama API.
            model (str): The name of the AI model to use.
            messages (List[Message]): The conversation history.
            prompt (str | None, optional): The initial prompt for the AI model.
                Defaults to None.
            image (str | None, optional): The path to the image file.
                Defaults to None.
        """
        ...

    def __repr__(self) -> str: ...

    def add_message(self, message: Message) -> None:
        """
        Append a message to the response that will be sent to the LLM in the chat mode.
        :param message: The message to add.
        """

    def add_response(self, response: LLMResponse) -> None:
        """
        Append a response to the response that will be sent to the LLM in the chat mode.
        :param response:
        """


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

    def cost(self) -> float:
        """
        Calculate cost of the response.

        The cost is calculated based on the model used and the number of tokens in the response.

        :return: The cost of the response in dollars.
        """
        ...


def send(request_body: AnthropicRequest | OpenAIRequest | OllamaRequest) -> LLMResponse:
    """
    Send prepared LLM Request
    :param request_body: AnthropicRequest or OpenAIRequest or OllamaRequest
    :return: LLMResponse
    """


def count_tokens(request_body: AnthropicRequest | OpenAIRequest) -> int:
    """
    Count tokens
    :param request_body: AnthropicRequest or OpenAIRequest
    :return: int
    """


def chat(request_body: OllamaRequest) -> LLMResponse:
    """
    Send prepared LLM Request
    :param request_body: OllamaRequest
    :return: LLMResponse
    """


def send_with_model(request_body: OpenAIRequest, model: type[BaseModel]) -> "GoldenAIParsedResponse":
    """
    Send prepared OpenAI Request with pydantic model
    :param request_body: OpenAIRequest
    :param model: pydantic model you want to parse from LLM response
    :return: GoldenAIParsedResponse
    """
