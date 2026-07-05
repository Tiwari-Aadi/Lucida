import os
import json
import re
from typing import List, Dict, Any

from groq import Groq


SYSTEM_PROMPT = """You are a structured curriculum parser.
Given a lesson, break it into ordered concept chunks.
Return ONLY a valid JSON array - no markdown, no explanation, no code fences.

Each element must follow this exact schema:
{
  "order": <int starting at 1>,
  "text": "<verbatim excerpt or paraphrase from the lesson>",
  "concepts": ["<concept name introduced in this chunk>"],
  "prerequisites": ["<concept name required to understand this chunk>"]
}

Rules:
- Every concept name used in "prerequisites" must appear in a previous chunk's "concepts".
- The first chunk must have an empty "prerequisites" list.
- Use concise concept names (1-5 words).
- Cover the full lesson without gaps."""


def parse_lesson(text: str) -> List[Dict[str, Any]]:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this lesson into concept chunks:\n\n{text}"},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps them anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    chunks = json.loads(raw)

    # Validate basic structure
    for i, chunk in enumerate(chunks):
        if "order" not in chunk:
            chunk["order"] = i + 1
        if "text" not in chunk:
            raise ValueError(f"Chunk {i} missing 'text' field")
        if "concepts" not in chunk:
            chunk["concepts"] = []
        if "prerequisites" not in chunk:
            chunk["prerequisites"] = []

    return chunks
