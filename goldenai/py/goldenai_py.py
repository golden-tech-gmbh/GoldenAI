import os

from typing import List, Dict, TYPE_CHECKING

from openai import OpenAI, AzureOpenAI
from openai.types import Reasoning
from openai.types.responses import ParsedResponse
from pydantic import BaseModel

from .struct import GoldenAIParsedResponse
from ..goldenai import OpenAIRequest


def send_with_model(request_body: OpenAIRequest, model: type[BaseModel]) -> "GoldenAIParsedResponse":
    client = None
    if request_body.endpoint:
        if "azure" in request_body.endpoint:
            client: "AzureOpenAI" = AzureOpenAI(
                api_version="2025-03-01-preview",
                azure_endpoint=request_body.endpoint,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            )
    if not client:
        client: "OpenAI" = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            # base_url=request_body.endpoint,
        )

    inputs: List[Dict] = request_body.msg_to_list_hashmap()
    reasoning: "Reasoning" = Reasoning(
        effort=request_body.reasoning.effort,
        summary=request_body.reasoning.summary,
    )

    response: "ParsedResponse" = client.responses.parse(
        model=request_body.model,
        input=inputs,
        text_format=model,
        instructions=request_body.instructions,
        max_output_tokens=request_body.max_output_tokens,
        reasoning=reasoning if "gpt-5" in request_body.model else None,
    )

    goldenai_response: "GoldenAIParsedResponse" = GoldenAIParsedResponse.from_parsed_response(response)

    return goldenai_response
