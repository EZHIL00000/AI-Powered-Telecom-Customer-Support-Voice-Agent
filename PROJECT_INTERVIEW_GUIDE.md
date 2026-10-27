# Lauki Phones Voice Agent — Project & Interview Guide

This guide explains the repository as it is implemented, including the document-ingestion pipeline, the terminal and live-voice workflows, the tools and sample data, and important limitations to discuss honestly in an interview.

> **Project in one sentence:** A Python customer-support assistant for a fictional telecom provider that accepts text or live audio, uses a Groq-hosted language model with LangChain tools, and retrieves answers from locally indexed PDF documentation with embeddings and Qdrant.

## 1. What problem does it solve?

The project demonstrates a conversational support agent for **Lauki Phones**. It can:

- Answer questions about plans, pricing, billing, recharges, and network coverage.
- Look up a sample customer's balance and usage from an in-memory mock database.
- Recommend a plan based on stated usage.
- Create a simulated support escalation.
- Handle text through a terminal loop or speech through a LiveKit voice session.

This is a learning/demo system rather than a production telecom integration. Lauki Phones, customer records, network status, and support tickets are sample data or simulated behavior; there is no live carrier account system, billing API, CRM, or ticketing service connected.

## 2. Technology at a glance

| Concern | Implementation |
|---|---|
| Language/runtime | Python 3.12 or newer |
| Voice session and real-time media | LiveKit Agents |
| Speech recognition | LiveKit inference using `deepgram/nova-3`, language `multi` |
| Speech synthesis | LiveKit inference using `inworld/inworld-tts-2`, voice `Ashley` |
| Agent and tool loop | LangChain `create_agent` and `@tool` |
| Chat model | `ChatGroq`, configured by `GROQ_MODEL`/`groq_model` (default `openai/gpt-oss-120b`) |
| PDF parsing/chunking | Docling `DocumentConverter` and `HierarchicalChunker` |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector search | Local persistent Qdrant collection, cosine distance |
| Configuration | Pydantic Settings reading `.env` and environment variables |
| Python packaging | `pyproject.toml`; a `uv.lock` file is present in the working tree |

The versions recorded in the local lock file include LiveKit Agents 1.8.8, LangChain 1.4.7, LangChain-Groq 1.1.3, Qdrant client 1.19.1, Sentence Transformers 6.2.0, and Docling 2.141.0. Treat these as the versions in this checkout's lock file, not as minimum requirements from `pyproject.toml`.

## 3. Repository map

| Path | Role |
|---|---|
| [`config.py`](./config.py) | Defines settings, defaults, required credentials, `.env` loading, and environment-variable setup for LiveKit/Groq. |
| [`ingest.py`](./ingest.py) | Parses PDFs, creates hierarchical text chunks, embeds them, and rebuilds the local Qdrant collection. |
| [`langchain_agent/tools.py`](./langchain_agent/tools.py) | Initializes the embedding model and Qdrant client, contains the mock data and six tools, defines the system prompt, and builds the LangChain agent. |
| [`langchain_agent/rag.py`](./langchain_agent/rag.py) | Shared vector retrieval/context helpers plus a standalone retrieve-and-generate RAG pipeline and CLI. |
| [`langchain_agent/main.py`](./langchain_agent/main.py) | Interactive text-only terminal chat loop. |
| [`langchain_agent/__init__.py`](./langchain_agent/__init__.py) | Re-exports the agent graph and its shared Qdrant/embedder objects. |
| [`voice_agent/main.py`](./voice_agent/main.py) | LiveKit server/session entry point, speech pipeline configuration, greeting, and streaming filter. |
| [`docs/lauki-plans.pdf`](./docs/lauki-plans.pdf) | Plan prices, allowances, benefits, and plan-selection guidance. |
| [`docs/lauki-billing-faq.pdf`](./docs/lauki-billing-faq.pdf) | Billing, payment, recharge, late-fee, refunds, and plan-change FAQs. |
| [`docs/lauki-network-coverage.pdf`](./docs/lauki-network-coverage.pdf) | City-level coverage descriptions, weak-signal steps, and 5G rollout information. |
| [`data/qdrant/`](./data/qdrant) | Local Qdrant persistence files; generated index data, not application source. |
| [`pyproject.toml`](./pyproject.toml) | Package metadata, Python floor, runtime dependencies, and wheel package list. |
| [`uv.lock`](./uv.lock) | Resolved dependency versions in this checkout. |
| [`README.md`](./README.md) | Short setup instructions and example customer prompts. |
| [`.gitignore`](./.gitignore) | Excludes environments, caches, local data, secrets, PDFs except those in `docs/`, and `*.lock` files. |

There are no application test files in the repository. The `.env` file is intentionally not reproduced here; never put its credential values into interview notes or source control. The README refers to `.env.example`, but that file is not present in this checkout.

## 4. High-level architecture

```text
                     ┌─────────────────────────┐
Text user ──────────▶│ langchain_agent/main.py │
                     └────────────┬────────────┘
                                  │ invoke(messages)
                                  ▼
                     ┌─────────────────────────┐
                     │ LangChain agent graph  │◀──────────────┐
                     │ Groq chat model         │               │
                     └───────┬─────────┬───────┘               │
                             │ tool call│ final answer          │
                ┌────────────┘         └─────────────▶ terminal│
                ▼                                             │
       Six LangChain tools                                    │
        ├─ local mock account/plan/network data               │
        ├─ vector search ──▶ local Qdrant ◀── PDF ingestion   │
        └─ simulated escalation                               │
                                                              │
Audio user ──▶ LiveKit room/session ──▶ STT ──▶ same agent graph
                                            │
                                            └─▶ TTS ──▶ audio reply
```

The source of truth differs by question type:

- **Plans, billing, and coverage prose:** PDF documents indexed in Qdrant and surfaced through `search_plans`.
- **Account balance, plan lookup/recommendation, and the tool's network summary:** Python dictionaries in `tools.py`.
- **Conversation/reasoning:** Groq model and the agent's system prompt.
- **Voice transport and speech:** LiveKit session plus its STT/TTS inference services.

## 5. End-to-end workflows

### A. Prepare the knowledge base (offline ingestion)

Run `python ingest.py` after configuring the project.

1. Read `settings.docs_dir` (default `docs`) and find sorted `*.pdf` files.
2. For each PDF, Docling's `DocumentConverter` parses the file into a structured document.
3. `HierarchicalChunker(max_characters=settings.chunk_size)` splits it into chunks (default maximum is 500 characters). `convert_chunk` retains headings as a breadcrumb before the chunk body so a retrieved passage keeps some section context.
4. Sentence Transformers encodes all chunk text with the configured embedding model (default `all-MiniLM-L6-v2`).
5. Qdrant creates/recreates the collection named `telecom_docs`, using the embedding model's reported vector dimension and cosine distance.
6. Each point is stored with the vector and payload fields `headings`, `content`, and `chunk_text`.

**Important operational detail:** ingestion uses `recreate_collection`, so each successful ingestion run replaces the existing collection rather than incrementally appending. Individual PDF failures are printed and skipped; if other PDFs succeed, the index may be rebuilt from only those successful documents. Ingestion runs before the agent in the README because agent startup expects the collection to exist.

### B. Answer a text question

Run `python -m langchain_agent.main`.

1. The CLI reads one line at a time, ignores blanks, and exits on `exit`, `quit`, or end-of-file.
2. It appends a LangChain `HumanMessage` to the conversation's `messages` list.
3. It calls `lauki_agent.invoke({"messages": messages})`.
4. The LangChain agent sends the conversation and system prompt to `ChatGroq` at temperature `0.0`. The model may answer directly or request one of the registered tools.
5. If it calls a tool, LangChain executes it, adds the tool result to the graph state, and gives the model an opportunity to produce a user-facing answer.
6. The CLI saves the returned messages for the next turn and prints the last message.

The terminal conversation state exists only in the Python process; there is no durable conversation store.

### C. Answer a live voice question

Run `python -m voice_agent.main console` for the documented local console workflow.

1. LiveKit Agents starts `AgentServer`; a room job invokes the `entrypoint`.
2. `AgentSession` is configured with Deepgram speech recognition, multilingual language mode, Inworld speech synthesis, voice `Ashley`, turn detection, and fixed endpointing delays (`0.5` to `6.0` seconds).
3. `session.start` connects `LaukiAgent` to the room. `LaukiAgent` adapts the same LangChain graph through `lk_langchain.LLMAdapter`.
4. The `SpeechOnlyGraph` wrapper filters `ToolMessage` chunks emitted by the graph. Tool output is therefore used by the model but is not itself spoken as though it were the assistant's final reply.
5. The agent generates an initial greeting and asks how it can help.
6. For later turns, LiveKit handles speech-to-text, invokes the agent/tool loop, and speaks the assistant output through TTS.

The voice model names are currently hardcoded in `voice_agent/main.py`. Although `config.py` defines voice model settings, that entry point does not read those settings.

### D. Retrieve a document-grounded answer

There are two closely related RAG paths:

**Tool path used by the main agent**

1. The model selects `search_plans(query)`.
2. `retrieve` encodes the query and searches the configured Qdrant collection for the five nearest points by default.
3. `build_context` joins each `chunk_text` into a context string.
4. The tool returns that context to the agent graph; the main agent model then formulates the spoken/text answer.

**Standalone RAG path in `langchain_agent/rag.py`**

1. `rag(query)` performs the same embedding, Qdrant search, and context assembly.
2. If no chunks are found, it returns a fixed “couldn't find” answer and empty context.
3. Otherwise it sends context and the question to a separate `ChatGroq` call at temperature `0.2`.
4. Its prompt restricts answers to the retrieved context, asks the model to admit when context is insufficient, cite a plan/section, keep answers to one or two short sentences, and format prices with ₹.

This standalone function can be tried with `python -m langchain_agent.rag "your question"`. It is not the direct path used by `search_plans`: that tool returns retrieved context, and the outer agent supplies the final answer.

## 6. The six tools

| Tool | What it does | Data/behavior |
|---|---|---|
| `search_plans(query)` | Retrieves relevant PDF chunks for plans, pricing, validity, comparisons, billing, and coverage questions. | Local Qdrant + Sentence Transformer; top five chunks. |
| `check_account_balance(phone_number)` | Validates a 10-digit numeric string and returns balance/bill, plan, and usage. | Three hardcoded sample customer records; no real account lookup. |
| `check_network_status(region="default")` | Returns 4G/5G labels and an overall coverage percentage. | Five city entries (Hyderabad, Bangalore, Delhi, Mumbai, Pune) plus a generic fallback. |
| `get_plan_recommendation(usage_type, data_gb=0, voice_min=0)` | Chooses a sample plan from a broad usage type or data requirement. | Uses hardcoded rules and plan data; `voice_min` is accepted but not used. |
| `get_plan_details(plan_name)` | Returns price, validity, allowances, and benefits for a named plan. | Case-insensitive lookup in the mock plans dictionary. |
| `escalate_to_support(reason)` | Returns a ticket-like confirmation and logs the request. | Simulated only: no persisted ticket or actual representative handoff. |

### Sample plan catalogue

The plan PDF describes these plans. The Python mock plan dictionary contains matching baseline details:

| Plan | Price / validity | Data | Voice | SMS | Example benefits |
|---|---:|---:|---:|---:|---|
| Lauki Lite 500MB | ₹99 / 7 days | 500 MB | 100 min | 50 | Pay-as-you-go after limit; no commitment. |
| Lauki Basic 1GB | ₹199 / 30 days | 1 GB | 300 min | 100 | Up to 500 MB rollover; free incoming calls while roaming in India. |
| Lauki Premium 2GB | ₹449 / 30 days | 2 GB | 300 min | 100 | Data rollover, priority support, one month of Netflix Mobile. |
| Lauki Elite 5GB | ₹899 / 30 days | 5 GB | 1,000 min | 500 | Unlimited data rollover, priority support, Netflix and Amazon Prime Video. |

These are fictional example prices and benefits. In a real service, prices and account entitlements should come from a governed, current backend rather than duplicated static dictionaries and PDFs.

### Sample network summary

The `check_network_status` tool uses the following sample figures: Hyderabad 99.5% (4G excellent, 5G available), Bangalore 99.8% (excellent, available), Delhi 98.2% (good, limited), Mumbai 99.7% (excellent, available), and Pune 97.8% (good, available). Other names use a generic 95% / good 4G / 5G coming-soon fallback.

The network PDF gives more nuanced neighborhood-level descriptions and cautions that coverage varies by building, terrain, and tower distance. The structured tool does not use those neighborhood-level details; it only returns the mock city record.

## 7. Configuration and startup

`config.py` defines a Pydantic `Settings` model. Settings are case-insensitive, read `.env`, and ignore extra keys. LiveKit URL/key/secret and the Groq API key are required. Other settings have defaults, including:

- `groq_model`: `openai/gpt-oss-120b`
- `docs_dir`: `docs`
- `embedding_model`: `all-MiniLM-L6-v2`
- `qdrant_path`: `./data/qdrant`
- `collection_name`: `telecom_docs`
- `chunk_size`: `500`

After settings load, the module sets LiveKit and Groq environment variables with `os.environ.setdefault`. Because settings are instantiated at module import, valid configuration must be available before importing the agent.

The normal setup sequence from the README is:

```powershell
uv sync
# Create/configure .env with LiveKit and Groq credentials.
python ingest.py
python -m voice_agent.main console
```

For terminal text chat, use:

```powershell
python -m langchain_agent.main
```

For stand-alone RAG experimentation:

```powershell
python -m langchain_agent.rag "What plans are available?"
```

The README says to copy `.env.example`, but no example file is in this checkout. Create a local `.env` with the required credential names without committing it; `.gitignore` excludes `.env`.

## 8. Important design choices

### Why use RAG?

Plan, billing, and network facts live in external documents and can change independently from the Python control logic. RAG converts document chunks into embeddings, retrieves semantically similar text for a question, and gives those passages to a language model. This is more appropriate than asking the model to rely on parametric memory for fictional or organization-specific policies. The standalone RAG prompt explicitly says not to invent facts when context is insufficient.

### Why a vector database?

Embedding similarity supports paraphrased questions (for example, “How can I pay my bill?” versus wording in a PDF about accepted payment methods). Qdrant stores each vector alongside the chunk payload, so search returns both relevance and the source text used to answer.

### Why separate tools and documents?

Tools are suited to structured, action-like operations such as looking up a mock account, checking a city status, or simulating escalation. RAG is suited to finding passages in longer explanatory PDFs. The language model chooses which operation to use based on the descriptions and system prompt.

### Why the streaming filter?

Tool results are internal evidence for the model, not necessarily a natural response to read aloud. `SpeechOnlyGraph` removes `ToolMessage` stream items before the LiveKit adapter emits audio, while leaving the assistant's response stream intact.

## 9. Limitations and interview-ready improvement ideas

Be clear about these boundaries if asked whether the project is production-ready:

1. **Mock business data:** Accounts, plans, and network status are hardcoded in `tools.py`; records are not updated from a live backend.
2. **Simulated escalation:** The ticket ID is derived from Python's built-in `hash(reason)`, printed to logs, and not stored. Python hash randomization means it is not a durable ticket identifier across processes. The two-hour callback promise is not backed by a service.
3. **Possible recommendation defect:** For `usage_type="heavy"`, the recommendation string is `"Lauki Premium 2GB or Elite 5GB"`. That entire string is not a key in `MOCK_PLANS_DB`, so the detailed plan lookup falls through to a generic message. Also, `voice_min` is unused and explicit recommendations consider only `data_gb`.
4. **Repeated configuration:** The voice agent hardcodes STT/TTS model settings instead of using corresponding `Settings` fields.
5. **Ingestion replacement and partial success:** Re-running ingestion deletes/recreates the collection; skipped files can result in an index missing some source docs. A safer production process would validate the full corpus, stage a versioned collection, and atomically switch it.
6. **Startup coupling:** Importing `langchain_agent.tools` loads the embedding model and opens Qdrant immediately. Missing local model files, an absent collection, or database errors can prevent application startup. The module intentionally raises a clear error if its collection cannot be opened.
7. **Local vector persistence:** Qdrant is file-backed at `./data/qdrant`; it is convenient for a demo but is not a managed, multi-instance production deployment.
8. **No automated tests:** No test files were found. Tool rules, ingestion behavior, voice filtering, and fallback cases should have unit/integration coverage.
9. **Potential source mismatch:** PDFs and the mock dictionaries both encode some plan/network facts. They can drift unless one source is authoritative and updates are validated.
10. **Secrets and observability:** Credentials need a secret manager in production. Logs should avoid exposing personal data, and production behavior needs tracing, metrics, latency/error reporting, and controlled escalation.

Reasonable next steps would include typed service interfaces for account/plan/network backends, durable ticket creation, dependency-injected model/vector clients, safer versioned ingestion, tests for each tool and RAG fallback, and evaluation sets for factual accuracy and voice latency.

## 10. Interview questions and concise answers

### “Walk me through the project.”

> “It is a Python voice-support demo for a fictional telecom. LiveKit manages the audio session and speech services. The session adapts a LangChain agent backed by Groq. That agent can call six tools: document search, mock account lookup, network lookup, plan recommendation, plan details, and simulated escalation. The documents are ingested with Docling, embedded with Sentence Transformers, and stored in local Qdrant. There is also a terminal chat mode that uses the same agent graph.”

### “What happens when a customer asks about plan pricing?”

> “The agent prompt directs plan questions to `search_plans`. The tool embeds the query with the same Sentence Transformer model used for document chunks, asks Qdrant for the five nearest vectors, and returns their text. The agent model uses that tool result to create a short user-facing answer. The separate `rag()` helper can also generate a context-grounded answer directly.”

### “What is an embedding, and why cosine distance?”

> “An embedding is a numeric vector representing text semantics. The query and document chunks are embedded in the same space so vector similarity can retrieve relevant passages even when the wording differs. The collection uses cosine distance to compare vector direction/similarity; the vector size is read from the embedding model rather than hardcoded.”

### “How is audio connected to the agent?”

> “LiveKit starts a room session with STT, turn detection/endpointing, and TTS. Its LangChain adapter wraps the same graph used by terminal chat. The graph can call tools, and the `SpeechOnlyGraph` wrapper filters tool-message tokens so internal tool output is not spoken directly; the final assistant response is synthesized.”

### “What data persists?”

> “The Qdrant vectors persist locally under `data/qdrant`. The account and plan dictionaries live only in process memory. The terminal conversation is in a local messages list. Escalations are printed to logs but not stored in a ticket system.”

### “How does the agent avoid making up policy facts?”

> “The standalone RAG prompt says to answer only from retrieved context and admit when context is insufficient. The main agent prompt says not to invent plan, price, or coverage facts and to use the search tool. Those instructions reduce hallucination risk, but they are not a formal guarantee; production would add source citations, validation, and evaluations.”

### “What would you improve first?”

> “I would replace mock account/plan/network data with authenticated service APIs, make escalations create durable tickets, fix and test heavy-plan recommendations, make voice settings use configuration, add safe staged/reproducible ingestion, and add tests and factual/latency evaluation. I would also move credentials to a secret manager and add structured observability.”

### “What is the biggest current correctness issue?”

> “The heavy recommendation path constructs a combined plan label that is not present in the plan dictionary, so it cannot return the full plan details. The recommendation tool also accepts `voice_min` but does not use it. I would correct the selection model and add tests for usage boundaries.”

## 11. Quick study checklist

- Be able to explain the distinction between **ingestion time** (PDF → chunks → embeddings → Qdrant) and **question time** (query embedding → nearest chunks → LLM answer).
- Remember the two user interfaces: text CLI and LiveKit voice session; both share the same LangChain graph.
- Know which facts are **RAG-backed** and which are **hardcoded mock data**.
- Explain the six tools and the purpose of the `SpeechOnlyGraph` stream filter.
- State the honest production gaps: no real account/ticket integrations, no test suite, local Qdrant, and simulated data.
- Mention concrete code findings rather than claiming integrations that are not present.

## 12. Source files to review before an interview

For a short code walkthrough, read these in order:

1. [`voice_agent/main.py`](./voice_agent/main.py) — live call lifecycle and speech configuration.
2. [`langchain_agent/tools.py`](./langchain_agent/tools.py) — agent prompt, tools, and sample business data.
3. [`ingest.py`](./ingest.py) — PDF-to-vector indexing.
4. [`langchain_agent/rag.py`](./langchain_agent/rag.py) — retrieval and standalone generation.
5. [`config.py`](./config.py) and [`pyproject.toml`](./pyproject.toml) — configuration and dependencies.
