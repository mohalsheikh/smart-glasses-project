from __future__ import annotations

from openai import OpenAI

# One shared OpenAI client instance across all brain modules.
client = OpenAI()
