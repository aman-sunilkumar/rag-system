import os
import yaml
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("Missing ANTHROPIC_API_KEY in your .env file")

client = Anthropic(api_key=api_key)

with open("configs/prompts.yaml", "r") as f:
    PROMPTS = yaml.safe_load(f)

def format_sources(chunks):
    return "\n\n".join(
        f"[source_{i+1}] (from {c['metadata']['file']}, "
        f"page {c['metadata'].get('page', 'n/a')}):\n{c['text']}"
        for i, c in enumerate(chunks)
    )

def generate(question: str, chunks: list) -> str:
    prompt = PROMPTS["answer_prompt"].format(
        sources=format_sources(chunks),
        question=question
    )

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return msg.content[0].text
