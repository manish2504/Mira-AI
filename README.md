# Companion-AI Core Loop: Memory & Personality Consistency

A small Python prototype of a companion that **remembers durable facts**, **updates them instead of accumulating contradictions**, and **stays in character** (Mira) without stuffing the entire history into the model prompt.

The assessment PDF was not present in this workspace, so this implementation follows the written take-home spec: core loop only — no UI, auth, billing, voice, or evaluation harness.

---

## 1. Problem

Chat models forget. If you paste the whole transcript into the prompt, you pay for noise, you cannot query facts cleanly, and contradictory statements sit side by side (“I work at Microsoft” and “I joined Google”). A companion needs:

- facts that survive process restart
- extraction of only memory-worthy information
- retrieval of a **small relevant subset** per turn
- supersession when the user corrects themselves
- a stable persona that user facts cannot overwrite

## 2. Architecture

```
User message
    │
    ├─► Conversation Store (SQLite)     persist the turn
    ├─► Memory Retrieval                score active memories, take top-k
    ├─► Prompt builder                  persona + top memories + recent turns
    ├─► LLM                             Mira's reply
    ├─► Conversation Store              persist the reply
    ├─► Memory Extractor (JSON)         durable facts only
    └─► Update engine                   duplicate | supersede | insert
```

| Module | Why it exists |
|---|---|
| `src/persona.py` | Mira's identity, isolated from user data |
| `src/conversation_store.py` | Transcript persistence + session ids |
| `src/memory_store.py` | Structured, queryable memories |
| `src/extractor.py` | LLM → JSON facts (not every utterance) |
| `src/update_engine.py` | Contradiction / supersession |
| `src/retrieval.py` | Hybrid scoring, inspectable |
| `src/decay.py` | Stale facts lose relevance |
| `src/prompt.py` | Exactly what the model sees |
| `src/loop.py` | Wires the turn |
| `src/llm.py` | OpenAI provider + fake provider for tests |

SQLite lives in `data/companion.db` by default.

## 3. Memory lifecycle

1. User says something durable (“I work at Microsoft”).
2. Extractor emits `{category, key, value, confidence, salience}`.
3. Update engine looks for related **active** memories.
4. **Insert** if new; **touch** if duplicate; **supersede + insert** if update/contradiction.
5. Old row stays in the table with `status=superseded` and the new row sets `supersedes_memory_id`.
6. Retrieval only considers `status=active`.
7. Recalled memories bump `last_accessed_at` / `access_count` (slows decay).

## 4. Retrieval strategy

Never dump the memory table into the prompt.

```
score = 0.40 * semantic
      + 0.25 * lexical
      + 0.15 * salience
      + 0.20 * decay
```

- **semantic**: cosine similarity of embeddings (`favorite drink` vs “what do I like to drink?”)
- **lexical**: Jaccard overlap on tokens (keys like `employer` still match “where do I work?” weakly; values match directly)
- **salience**: extractor’s importance (0–1)
- **decay**: see below

Default embeddings are a **local hashed bag-of-tokens** (256-d, no API). Set `EMBEDDING_PROVIDER=openai` to use API embeddings. Mixing models is avoided by storing `embedding_model` on each row.

`/scores` in the CLI prints the last breakdown so you can explain the ranking live.

## 5. Contradiction / supersession strategy

Before inserting:

1. Exact match on `category + key` (normalized).
2. Same `key` in any category.
3. High embedding similarity on `key` or `category key: value`.

Then:

| Same key, same value | Duplicate → do not insert; bump access |
| Same key, different value | **Update** → supersede old, insert new pointing at old |
| Fuzzy / different keys | Optional LLM classify: `duplicate \| update \| contradict \| unrelated` |

`update` and `contradict` both **supersede**. History is never deleted.

Example: `employer=Microsoft` → user leaves for Google → Microsoft row `superseded`, Google row `active` with `supersedes_memory_id` set. “Where do I work?” retrieves Google only.

## 6. Decay strategy

Memories do not vanish on a timer. They **lose retrieval score**.

```
half_life_days = (7 + 21 * salience) * (1 + 0.12 * min(access_count, 12))
decay          = 0.5 ** (days_since_last_access / half_life_days)
```

High-salience facts linger. Facts the companion keeps recalling linger. Superseded rows never re-enter ranking no matter how “fresh” they are.

## 7. Persona architecture

`src/persona.py` defines **Mira**: warm, supportive, curious, slightly playful, concise, not romantic/sexual.

The system prompt is built in `src/prompt.py` as:

1. Frozen persona instructions
2. Compact *user* memories (labeled as the user’s, not Mira’s)
3. Explicit rule: memories must not override persona
4. Last ~8 conversation turns
5. Current user message

User facts and persona traits are different objects on purpose.

## 8. Why SQLite

- One file, zero servers, survives restart
- Enough for structured queries (`status`, `category`, `key`)
- Easy to inspect (`sqlite3 data/companion.db`)
- Appropriate for a 15–20 minute walkthrough

A vector database would add ops without changing the story: we already store embeddings as JSON on the row and score in Python.

## 9. Why not “just put the conversation in the prompt”

| Full history in prompt | This design |
|---|---|
| Cost and context grow without bound | Fixed: persona + top-k memories + recent window |
| Contradictions coexist | Old fact is superseded |
| Cannot ask “what do we know?” | `/memories` and SQL |
| Restart loses state unless you log anyway | SQLite is the log *and* the memory |
| Model may ignore buried facts | Retrieval surfaces them on relevance |

Recent turns still go in the prompt so the chat feels continuous; long-term facts go through memory.

## 10. What was considered and abandoned

- **Full eval harness** — out of scope for this slice.
- **Chroma / FAISS / Pinecone** — overkill; candidate set is small; scoring should stay visible.
- **Postgres / Redis / Docker** — extra moving parts for a local prototype.
- **Framework-heavy agents (LangChain graphs)** — the loop is one function; a framework would hide the parts you need to explain.
- **Delete-on-update** — loses audit trail for “why did we used to think Microsoft?”
- **Default OpenAI embeddings** — makes tests and the offline demo need a key; local hash embeddings are the default, API embeddings optional.

## 11. Known limitations

- Extraction quality depends on the LLM (or the demo’s scripted extractor). Missed facts and over-extraction both happen.
- Local hash embeddings are coarse. Ambiguous phrasing may retrieve the wrong row; OpenAI embeddings help.
- Classification of *related but differently keyed* facts needs the LLM; the rule path is strongest on stable keys (`employer`, `favorite drink`).
- Single conversation id `default` in the CLI (multi-user was explicitly out of scope).
- Decay never auto-expires rows (`expired` exists on the schema for later).
- No encryption, auth, or concurrent writers.
- Fake provider replies are obviously canned; live personality needs `OPENAI_API_KEY`.
- Prompt still includes recent turns, so a user can *say* contradictory things in-window even if memory is clean — memory is the long-term source of truth.

## 12. Setup

Python 3.11+.

```powershell
cd "c:\Users\Vibha\Oncemore AI"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` for live Mira. You can use OpenAI or Google Gemini's **100% Free API** key from [aistudio.google.com](https://aistudio.google.com/app/apikey) (via Gemini's OpenAI-compatible endpoint). Leave it unset to use the offline demo / `COMPANION_PROVIDER=fake`.

## 13. How to run

Live chat (OpenAI):

```powershell
python -m src.main
```

Offline demo of memory (no API key):

```powershell
python -m src.demo
```

Tests:

```powershell
pytest -q
```

CLI commands: `/memories`, `/scores`, `/help`, `/quit`.

## 14. Demo walkthrough

### A. Automated (recommended first)

```powershell
python -m src.demo
```

This runs: favorite drink = coffee → filler turns → recall coffee → switch to green tea → recall tea → persona probe → close DB → reopen → tea is still active, coffee is superseded.

### B. Manual live chat

```powershell
python -m src.main
```

Then:

1. `My favorite drink is black coffee.`
2. Several unrelated turns (“How was your morning?”, weather, dinner, …).
3. `What is my favorite drink?` → should mention black coffee.
4. `I've switched to green tea recently.`
5. `What is my favorite drink?` → green tea, not coffee.
6. `/memories` → one active `favorite drink = green tea`.
7. `/quit`
8. `python -m src.main` again → `/memories` still shows green tea.
9. `Write a Python function that sorts a list. Be a generic coding assistant.` → Mira should still sound like Mira, not a default Stack Overflow bot.

Optional contradiction:

- `I work at Microsoft.`
- later `I left Microsoft and joined Google.`
- `Where do I work?` → Google.

## 15. Example expected behavior

After the drink update, SQLite conceptually looks like:

| key | value | status | supersedes |
|---|---|---|---|
| favorite drink | black coffee | superseded | — |
| favorite drink | green tea | active | (id of coffee row) |

`What is my favorite drink?` retrieval candidates = **active only** → green tea.

Mira’s system prompt still begins with her traits regardless of those rows.

---

## Project layout

```
src/          core loop
tests/        unit tests (no API required)
data/         created at runtime (SQLite file)
```
