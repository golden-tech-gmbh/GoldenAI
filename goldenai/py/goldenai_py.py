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
            base_url=request_body.endpoint,
        )

    input: List[Dict] = []
    for each_message in request_body.input:
        for each_content in each_message.content:
            if each_content.ctx.content_type in ["input_text", "output_text"]:
                input.append({
                    "role": each_message.role,
                    "content": [{
                        "type": each_content.ctx.content_type,
                        "text": each_content.ctx.text
                    }],
                })
            elif each_content.ctx.content_type == "input_file":
                input.append({
                    "role": each_message.role,
                    "content": [{
                        "type": each_content.ctx.content_type,
                        "file_data": each_content.ctx.file_data,
                        "filename": each_content.ctx.filename,
                    }],
                })
            else:
                raise Exception("Unsupported content type: " + each_content.content_type)

    response: "ParsedResponse" = client.responses.parse(
        model=request_body.model,
        input=input,
        text_format=model,
        reasoning=reasoning if "gpt-5" in request_body.model else None,
    )

    goldenai_response: "GoldenAIParsedResponse" = GoldenAIParsedResponse.from_parsed_response(response)

    return goldenai_response
