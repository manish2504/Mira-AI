from __future__ import annotations

from openai import OpenAI

from src.config import Settings
from src.conversation_store import ConversationStore
from src.db import connect
from src.embeddings import LocalHashEmbedder, OpenAIEmbedder
from src.extractor import MemoryExtractor
from src.llm import LLMProvider, build_provider
from src.loop import CompanionLoop
from src.memory_store import MemoryStore
from src.retrieval import MemoryRetriever
from src.update_engine import MemoryUpdateEngine


def build_embedder(settings: Settings):
    if settings.embedding_provider == "openai":
        if not settings.gemini_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires GEMINI_API_KEY")
        kwargs: dict = {"api_key": settings.gemini_api_key}
        if settings.gemini_base_url:
            kwargs["base_url"] = settings.gemini_base_url
        client = OpenAI(**kwargs)
        return OpenAIEmbedder(client, settings.openai_embedding_model)
    return LocalHashEmbedder()


def build_loop(
    settings: Settings,
    *,
    llm: LLMProvider | None = None,
    conversation_id: str = "default",
    session_id: str | None = None,
) -> tuple[CompanionLoop, object]:
    conn = connect(settings.database_path)
    conversations = ConversationStore(conn)
    memories = MemoryStore(conn)
    embedder = build_embedder(settings)
    provider = llm or build_provider(settings)
    retriever = MemoryRetriever(memories, embedder, settings)
    extractor = MemoryExtractor(provider)
    updater = MemoryUpdateEngine(
        memories,
        embedder,
        llm=provider,
        related_threshold=settings.related_similarity_threshold,
    )
    loop = CompanionLoop(
        settings=settings,
        conversations=conversations,
        memories=memories,
        retriever=retriever,
        extractor=extractor,
        updater=updater,
        llm=provider,
        conversation_id=conversation_id,
        session_id=session_id,
    )
    return loop, conn
