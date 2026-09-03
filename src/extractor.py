from __future__ import annotations

import json

from src.llm import LLMProvider
from src.models import ExtractedFact, ExtractionResult

EXTRACTOR_SYSTEM = """Extract durable memories about the USER from their latest message.

Return JSON: {"memories": [ ... ]}

Each item:
{
  "category": "preference|work|plan|relationship|goal|personal|event|correction",
  "key": "short spaced identifier, e.g. favorite drink, employer, last cricket match",
  "value": "canonical current fact",
  "confidence": 0.0-1.0,
  "salience": 0.0-1.0
}

IMPORTANT RULES FOR EXTRACTION:

1. FAVORITES vs GENERAL LIKES:
   - "My favorite drink is coffee" -> key: "favorite drink", value: "coffee"
   - "I like banana shake" -> key: "liked beverage", value: "banana shake"
   - "I love pizza" -> key: "liked food", value: "pizza"
   - ONLY use "favorite X" as the key when the user EXPLICITLY says "my favorite",
     "my fav", or "X is my favorite". General statements like "I like", "I enjoy",
     "I love" should use "liked X" as the key.
   - "I switched to tea" or "my new favorite is tea" -> key: "favorite drink"
     (explicit replacement language).

2. EVENTS and EXPERIENCES (category: "event"):
   - Sports results: key "last cricket match", value "got out on 0, LBW"
   - Travel: key "recent trip", value "visited Goa last weekend"
   - Life events: key "recent achievement", value "passed driving test"
   - Extract specific details like scores, how they got out, what happened.

3. PERSONAL DETAILS (category: "personal"):
   - Name, age, location, hobbies, pets, family details.
   - key: "name", value: "Manish"

4. WORK and EDUCATION (category: "work"):
   - Employer, role, school, degree.

5. CORRECTIONS (category: "correction"):
   - When the user explicitly corrects a prior statement: "Actually, I meant..."
   - Use the CORRECTED fact as the value with high salience.

Remember:
- Extract ALL substantive facts, not just the most obvious one.
- Sports match details, dismissals, scores are important personal events.
- Preferences, work/employment, plans, relationships, goals, personal context.

Do NOT remember:
- Greetings, small talk, questions asked TO the assistant, jokes.
- Ephemeral feelings unless they are a stated lasting preference.

If the user updates a fact ("I left Microsoft and joined Google"), extract the
CURRENT state (employer = Google) with high salience.

If nothing is worth storing, return {"memories": []}.
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
