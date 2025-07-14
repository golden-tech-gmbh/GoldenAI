"""
This is an example of how to use goldenai
- open goldenai repo
- prepare venv and install wheels
- Set PYTHONPATH to the root directory of goldenai and run
```sh
python examples/python/example.py
```
"""


def example_using_anthropic():
    from goldenai import Content, Message, AnthropicRequest, send, LLMResponse, count_tokens

    # # generate a content object
    # # alternatively, you can also:
    # from goldenai import TextContent
    # text = TextContent("Hello, Claude!")
    # content = Content(text)
    # # or
    # from goldenai import DocumentContent
    # doc = DocumentContent("white.jpg")
    # content = Content(doc)

    content = Content.from_text("Hello, Claude!")
    content2 = Content.from_text("What is color of this image?")
    content3 = Content.from_document("examples/python/white.jpg")

    # construct a message
    message = Message(content=[content, content2, content3])

    # contruct a request
    request = AnthropicRequest(model="claude-3-5-haiku-latest", messages=[message],
                               max_tokens=1024,  # optional
                               prompt="Please answer in Chinese"  # optional
                               )

    # count the tokens
    print(count_tokens(request))

    # send the request
    # model should be one of LLM.Anthropic or LLM.OpenAI, default LLM.Anthropic
    res: LLMResponse = send(request_body=request)

    print(res)
    print(res.cost())

    # you can add further response and messages to the request for chat with context
    request.add_response(res)
    content2 = Content.from_text("Please answer again in English")
    message2 = Message(content=[content2])
    request.add_message(message2)
    res2: LLMResponse = send(request_body=request)
    print(res2)


def example_using_openai():
    import os
    from goldenai import Content, Message, OpenAIRequest, send, count_tokens, LLMResponse

    content = Content.from_text("Hello, OpenAI!")
    content2 = Content.from_text("What does this document say?")
    # for OpenAI, you need to construct a document content with llm parameter
    # And for know, it is only possible to process pdf files
    content3 = Content.from_document("examples/python/test.pdf", llm="gpt-4.1-nano-2025-04-14")

    message = Message(content=[content, content2, content3])
    request = OpenAIRequest(model="gpt-4.1-nano-2025-04-14", messages=[message],
                            # no max_tokens option for OpenAI
                            prompt="Please answer in Chinese",  # optional
                            endpoint=f"https://{os.getenv('AZURE_RESOURCE_NAME')}.openai.azure.com/openai/v1/responses?api-version=preview"
                            # optional, here is an example using model provided by Azure, NOTE: only support 'responses' endpoint
                            )

    res = send(request)

    print(res)
    print(res.cost())

    # you can add further response and messages to the request for chat with context
    request.add_response(res)
    content2 = Content.from_text("Please answer again in English")
    message2 = Message(content=[content2])
    request.add_message(message2)
    res2: LLMResponse = send(request_body=request)
    print(res2)


def example_using_ollama():
    from goldenai import Content, Message, OllamaRequest, send

    content = Content.from_text("What is the color of this image?")
    # DO NOT CONSTRUCT DOCUMENT CONTENT IN OLLAMA, PASS IT DIRECTLY IN REQUEST
    # content = Content.from_document("white.jpg")  # WRONG USAGE! THIS WON'T WORK!
    message = Message(content=[content])
    request = OllamaRequest(url="http://10.8.0.1:11434", model="qwen2.5vl:latest", messages=[message],
                            prompt="Please answer in Chinese", image="examples/python/white.jpg")

    res = send(request)

    print(res)


def example_chat_ollama():
    from goldenai import Content, Message, OllamaRequest, send, LLMResponse, chat

    content = Content.from_text("What is your name?")
    message = Message(content=[content])
    request = OllamaRequest(url="http://10.8.0.1:11434", model="qwen2.5vl:latest", messages=[message])
    res: LLMResponse = send(request)  # the first request can be sent either in send mode or chat mode
    print(res)
    request.add_response(res)
    content2 = Content.from_text("Please answer again in Chinese")
    message2 = Message(content=[content2])
    request.add_message(message2)  # MUST USE add_message to append messages in chat mode
    res2: LLMResponse = chat(request)  # MUST USE chat to enable chat mode and understand the context
    print(res2)


if __name__ == "__main__":
    example_using_anthropic()
    example_using_openai()
    # example_using_ollama()
    # example_chat_ollama()
