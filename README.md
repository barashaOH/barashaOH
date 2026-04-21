## Local Agentic Cybersecurity Bot (Python + Ollama + LangChain + RAG)

A clean, local-first cybersecurity assistant you can run on your own machine.

### What this version improves
- Better structure and configurability (env-driven settings)
- Defensive guardrails to avoid offensive misuse
- One-shot mode (`--ask`) + interactive chat mode
- Ingestion reset option (`ingest.py --reset`)
- Cleaner setup workflow

---

## 1) Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

Pull local models:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

---

## 2) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## 3) Add cybersecurity docs and ingest

Place your docs in `knowledge/` (`.txt`, `.md`, `.pdf`) then run:

```bash
python ingest.py
```

Rebuild index from scratch:

```bash
python ingest.py --reset
```

---

## 4) Run bot

Interactive mode:

```bash
python agent.py
```

One-shot mode:

```bash
python agent.py --ask "Give me a ransomware triage checklist for first 60 minutes"
```

---

## 5) Environment configuration

Configure models and behavior in `.env`:

```env
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TEMPERATURE=0.1
RAG_TOP_K=4
```

---

## Files

- `agent.py` → agent orchestration, tools, CLI
- `ingest.py` → local document ingestion into Chroma
- `knowledge/` → your cybersecurity corpus
- `knowledge/sample_playbook.md` → starter document

---

## Safety

This bot is for **defensive cybersecurity** use cases: hardening, detection, response, and remediation.
