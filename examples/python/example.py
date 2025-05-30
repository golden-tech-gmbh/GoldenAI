def example_using_anthropic():
    from goldenai import Content, Message, AnthropicRequest, send, LLMResponse, count_tokens

    # generate a content object
    # alternatively, you can also:
    # from goldenai import TextContent
    # text = TextContent("Hello, Claude!")
    # content = Content(text)
    content = Content.from_text("Hello, Claude!")
    content2 = Content.from_text("What is the day today?")

    # construct a message
    message = Message(content=[content, content2])

    # contruct a request
    request = AnthropicRequest(model="claude-3-5-haiku-latest", max_tokens=1024, messages=[message],
                               prompt="Please answer in Chinese")

    # count the tokens
    print(count_tokens(request))

    # send the request
    # model should be one of LLM.Anthropic or LLM.OpenAI, default LLM.Anthropic
    res: LLMResponse = send(request_body=request)

    print(res)
    print(res.cost())


def example_using_openai():
    from goldenai import Content, Message, OpenAIRequest, send, count_tokens

    content = Content.from_text("Hello, Claude!")
    content2 = Content.from_text("What is the day today?")
    message = Message(content=[content, content2])
    request = OpenAIRequest(model="gpt-4.1-nano-2025-04-14", max_tokens=1024, messages=[message],
                            prompt="Please answer in Chinese")

    print(count_tokens(request))

    res = send(request)

    print(res)
    print(res.cost())


if __name__ == "__main__":
    example_using_anthropic()
    example_using_openai()
