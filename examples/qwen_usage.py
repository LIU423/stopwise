"""Qwen example using Alibaba Cloud Model Studio's compatible interface."""

import os

from openai import OpenAI

from stopwise import StopWise


client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    # This URL is region- and workspace-specific.
    base_url=os.environ["DASHSCOPE_BASE_URL"],
)

stopwise = StopWise(
    client=client,
    model=os.getenv("QWEN_MODEL", "qwen-plus"),
    transport="chat_completions",
    request_options={"extra_body": {"enable_thinking": False}},
)

messages = [
    {
        "role": "user",
        "content": "A 已经满足预算、性能和续航要求。我又确认了三次，结果都没有变化，还需要继续比较吗？",
    }
]

result = stopwise.analyze(messages)
print(result.action)
print(result.message)

