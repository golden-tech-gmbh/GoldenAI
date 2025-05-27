from goldenai import TextContent, AnthropicRequest, Message, Content, send

# text_content = TextContent(text="Hello, Claude!")
content = Content.from_text("Hello, Claude!")
message = Message(content=[content])
request = AnthropicRequest(model="claude-3-5-haiku-latest", max_tokens=1024, messages=[message],
                           prompt="Please answer in Chinese")

res = send(request)

print(res)
pass
