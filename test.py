from goldenai import TextContent, AnthropicRequest, Message, Content, get_response

# text_content = TextContent(text="Hello, Claude!")
# content = Content(text_content)
content = Content.from_text("Hello, Claude!")
message = Message(role="user", content=[content])
request = AnthropicRequest(model="claude-3-5-haiku-latest", max_tokens=1024, messages=[message])

res = get_response(request)

pass
