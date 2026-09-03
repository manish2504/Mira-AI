from __future__ import annotations

import json

from src.llm import LLMProvider
from src.models import ExtractedFact, ExtractionResult

EXTRACTOR_SYSTEM = """Extract durable memories about the USER from their latest message.

Return JSON: {"memories": [ ... ]}

Each item:
{
  "category": "preference|work|plan|relationship|goal|personal|correction",
  "key": "short snake or spaced identifier, e.g. favorite drink, employer",
  "value": "canonical current fact",
  "confidence": 0.0-1.0,
  "salience": 0.0-1.0
}

Remember:
- preferences, work/employment, important plans, relationships, recurring goals,
  stable personal context, explicit corrections.

Do NOT remember:
- greetings, small talk, one-off questions, jokes, assistant instructions,
  ephemeral feelings unless they are a stated lasting preference.

If the user updates a fact ("I left Microsoft and joined Google"), extract the
CURRENT state (employer = Google) with high salience. Optionally also extract
historical facts under a distinct key such as former employer.

If nothing is worth storing, return {"memories": []}.
Use consistent keys for the same kind of fact (favorite drink, not fav_bev).
"""


class MemoryExtractor:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def extract(self, user_message: str) -> list[ExtractedFact]:
        raw = self.llm.complete(
            [
                {"role": "system", "content": EXTRACTOR_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            json_mode=True,
            temperature=0.0,
        )
        return parse_extraction(raw)


def parse_extraction(raw: str) -> list[ExtractedFact]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extractor returned invalid JSON: {raw[:200]}") from exc
    result = ExtractionResult.model_validate(data)
    return result.memories
