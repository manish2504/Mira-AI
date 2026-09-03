from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from src.config import Settings
from src.conversation_store import ConversationStore
from src.extractor import MemoryExtractor
from src.llm import LLMProvider
from src.memory_store import MemoryStore
from src.models import Memory, ScoreBreakdown
from src.persona import Persona, get_persona
from src.prompt import build_chat_messages
from src.retrieval import MemoryRetriever
from src.update_engine import MemoryUpdateEngine


@dataclass
class TurnResult:
    reply: str
    user_message_id: str
    assistant_message_id: str
    extracted_count: int
    retrieved: list[ScoreBreakdown] = field(default_factory=list)
    new_memories: list[Memory] = field(default_factory=list)


class CompanionLoop:
    """The walkthrough-sized core: persist → retrieve → generate → extract → update."""

    def __init__(
        self,
        settings: Settings,
        conversations: ConversationStore,
        memories: MemoryStore,
        retriever: MemoryRetriever,
        extractor: MemoryExtractor,
        updater: MemoryUpdateEngine,
        llm: LLMProvider,
        persona: Persona | None = None,
        conversation_id: str = "default",
        session_id: str | None = None,
    ):
        self.settings = settings
        self.conversations = conversations
        self.memories = memories
        self.retriever = retriever
        self.extractor = extractor
        self.updater = updater
        self.llm = llm
        self.persona = persona or get_persona()
        self.conversation_id = conversation_id
        self.session_id = session_id or str(uuid.uuid4())
        self.last_retrieval: list[ScoreBreakdown] = []
        self.conversations.ensure_conversation(self.conversation_id)

    def handle_user_message(self, text: str) -> TurnResult:
        user_msg = self.conversations.add_message(
            self.conversation_id, self.session_id, "user", text
        )

        scores = self.retriever.retrieve(text)
        self.last_retrieval = scores
        recalled = self.retriever.memories_for(scores)

        recent = self.conversations.recent(
            self.conversation_id, self.settings.recent_turn_limit
        )
        prompt = build_chat_messages(self.persona, recalled, recent, text)
        reply = self.llm.complete(prompt, json_mode=False, temperature=0.5)

        assistant_msg = self.conversations.add_message(
            self.conversation_id, self.session_id, "assistant", reply
        )

        try:
            facts = self.extractor.extract(text)
        except (ValueError, Exception):
            facts = []
        created = self.updater.apply(
            facts,
            conversation_id=self.conversation_id,
            message_id=user_msg.id,
        )
        return TurnResult(
            reply=reply,
            user_message_id=user_msg.id,
            assistant_message_id=assistant_msg.id,
            extracted_count=len(facts),
            retrieved=scores,
            new_memories=created,
        )
