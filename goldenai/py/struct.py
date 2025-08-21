import copy
from openai.types.responses import ParsedResponse


class GoldenAIParsedResponse(ParsedResponse):

    def __str__(self) -> str:
        return str(self.output_text)

    @classmethod
    def from_parsed_response(cls, response: ParsedResponse) -> "GoldenAIParsedResponse":
        instance = copy.deepcopy(response)
        instance.__class__ = cls

        return instance

    def cost(self) -> float:
        if "gpt-4.1-nano" in self.model:
            input = 0.1
            output = 0.4
        elif "gpt-4.1-mini" in self.model:
            input = 0.4
            output = 1.6
        elif "gpt-4.1" in self.model:
            input = 2
            output = 8
        elif "gpt-5-nano" in self.model:
            input = 0.05
            output = 0.4
        elif "gpt-5-mini" in self.model:
            input = 0.25
            output = 2
        elif "gpt-5" in self.model:
            input = 1.25
            output = 10
        else:
            raise Exception("Unsupported model: " + self.model)

        input_tokens: int = self.usage.input_tokens
        output_tokens: int = self.usage.output_tokens

        return (input * input_tokens + output * output_tokens) / 1_000_000


__all__ = ["GoldenAIParsedResponse"]
