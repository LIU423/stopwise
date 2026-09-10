"""Minimal OpenAI example. Requires OPENAI_API_KEY."""

from openai import OpenAI

from stopwise import StopWise


client = OpenAI()
stopwise = StopWise(client=client, model="gpt-4o-mini")

messages = [
    {
        "role": "user",
        "content": (
            "I need a laptop under $1,500 with all-day battery life for my course. "
            "Model A meets those needs, but I haven't confirmed that it runs the "
            "required CAD package. Should I investigate that before buying?"
        ),
    }
]

result = stopwise.analyze(messages)
print(result.model_dump_json(indent=2))

