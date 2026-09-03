from src.extractor import parse_extraction
from src.models import ExtractedFact
from src.persona import get_persona
from src.prompt import build_chat_messages, format_memories
from tests.conftest import make_memory


def test_parse_extraction_structured_json():
    facts = parse_extraction(
        '{"memories":[{"category":"work","key":"Employer","value":"Google","confidence":0.9,"salience":0.8}]}'
    )
    assert facts == [
        ExtractedFact(
            category="work",
            key="employer",
            value="Google",
            confidence=0.9,
            salience=0.8,
        )
    ]


def test_parse_extraction_empty():
    assert parse_extraction('{"memories":[]}') == []


def test_prompt_keeps_persona_separate_from_user_memories(embedder):
    persona = get_persona()
    mem = make_memory(
        embedder, key="favorite drink", value="green tea", category="preference"
    )
    messages = build_chat_messages(persona, [mem], [], "What is my favorite drink?")
    system = messages[0]["content"]
    assert persona.name in system
    assert "green tea" in system
    assert "must not override your persona" in system
    assert messages[-1]["content"] == "What is my favorite drink?"
    assert format_memories([]) == "(none)"
