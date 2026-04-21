"""Local agentic cybersecurity assistant (Ollama + LangChain + RAG).

Features
- Local-only LLM via Ollama
- RAG against user-owned docs in ./knowledge (persisted in ./vectorstore)
- Tool-enabled behavior (RAG search + IOC extraction)
- Defensive guardrails against offensive misuse

Run interactive mode:
    python agent.py

Run one-shot mode:
    python agent.py --ask "How should I triage suspicious PowerShell activity?"
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "vectorstore"
DOCS_DIR = BASE_DIR / "knowledge"


@dataclass
class Settings:
    """Runtime configuration for the local cybersecurity assistant."""

    chat_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"
    temperature: float = 0.1
    top_k: int = 4

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
        )


class CyberSecLocalAgent:
    """Agent wrapper with RAG + helper tools for defensive cyber operations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.llm = ChatOllama(
            model=self.settings.chat_model,
            temperature=self.settings.temperature,
        )
        self.embeddings = OllamaEmbeddings(model=self.settings.embed_model)

        self.vectorstore = Chroma(
            collection_name="cybersec_knowledge",
            embedding_function=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR),
        )

        self.tools = [
            Tool(
                name="cybersec_rag_search",
                func=self.rag_search,
                description=(
                    "Search locally indexed cybersecurity documents. "
                    "Use for incident response, hardening, threat detection, "
                    "vulnerability remediation, and policy questions."
                ),
            ),
            Tool(
                name="ioc_quick_analyzer",
                func=self.analyze_iocs,
                description=(
                    "Extract IOC candidates from raw text/logs. "
                    "Returns IPs, domains, and likely file hashes."
                ),
            ),
            Tool(
                name="defensive_scope_checker",
                func=self.defensive_scope_checker,
                description=(
                    "Use when user asks potentially risky actions. "
                    "Input is a plain user request. Output tells whether to refuse "
                    "or proceed with defensive guidance."
                ),
            ),
        ]

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=6,
        )

    def rag_search(self, query: str) -> str:
        """Retrieve top-k cybersecurity passages from local vector DB."""
        docs = self.vectorstore.similarity_search(query, k=self.settings.top_k)
        if not docs:
            return (
                "No local knowledge found. Add files to ./knowledge and run: "
                "python ingest.py"
            )

        blocks = []
        for idx, doc in enumerate(docs, start=1):
            source = Path(doc.metadata.get("source", "unknown")).name
            snippet = doc.page_content[:1000].strip()
            blocks.append(f"[{idx}] {source}\n{snippet}")
        return "\n\n".join(blocks)

    @staticmethod
    def analyze_iocs(log_blob: str) -> str:
        """Extract simple IOC candidates from free-form text."""
        ipv4 = sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", log_blob)))
        domains = sorted(set(re.findall(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", log_blob)))
        hashes = sorted(set(re.findall(r"\b[a-fA-F0-9]{32,64}\b", log_blob)))

        return "\n".join(
            [
                "Quick IOC summary:",
                f"- IPv4: {', '.join(ipv4) if ipv4 else 'None'}",
                f"- Domains: {', '.join(domains) if domains else 'None'}",
                f"- Hashes: {', '.join(hashes) if hashes else 'None'}",
            ]
        )

    @staticmethod
    def defensive_scope_checker(user_text: str) -> str:
        """Apply lightweight policy checks to prevent harmful/offensive instructions."""
        blocked_keywords: Iterable[str] = (
            "exploit",
            "payload",
            "phishing kit",
            "credential stuffing",
            "sql injection",
            "xss attack",
            "ddos",
            "ransomware builder",
            "malware",
            "backdoor",
        )
        lowered = user_text.lower()
        if any(keyword in lowered for keyword in blocked_keywords):
            return (
                "Potentially offensive request detected. Refuse operational attack steps "
                "and provide safe alternatives (prevention, detection, hardening)."
            )
        return "Looks defensive or benign. Proceed with secure best-practice guidance."

    def ask(self, user_query: str) -> str:
        """Invoke the agent with a safety-first system directive."""
        system_hint = (
            "You are a local cybersecurity assistant focused on blue-team outcomes. "
            "Always prefer prevention, detection, incident response, compliance, and "
            "risk reduction. Decline offensive abuse and redirect to defensive advice. "
            "When domain knowledge is needed, use cybersec_rag_search. "
            "When logs are provided, use ioc_quick_analyzer."
        )
        prompt = f"{system_hint}\n\nUser question: {user_query}"
        return str(self.agent.run(prompt))


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local cybersecurity AI agent")
    parser.add_argument("--ask", type=str, default="", help="Run a one-shot query and exit")
    return parser


def run_interactive(agent: CyberSecLocalAgent, console: Console) -> None:
    console.print(
        Panel.fit(
            "Local Cybersecurity AI Agent\n[dim]Ollama + LangChain + RAG[/dim]",
            border_style="green",
        )
    )
    console.print("Type [bold]exit[/bold] or [bold]quit[/bold] to stop.\n")

    while True:
        query = console.input("[bold cyan]You> [/]").strip()
        if query.lower() in {"exit", "quit"}:
            console.print("[yellow]Session ended.[/]")
            break
        if not query:
            continue

        try:
            answer = agent.ask(query)
            console.print(Markdown(f"**Agent**\n\n{answer}"))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/] {exc}")
            console.print(
                "Check Ollama service + models, then run `python ingest.py` after adding docs."
            )


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    agent = CyberSecLocalAgent(settings=settings)
    args = build_cli().parse_args()
    console = Console()

    if args.ask:
        output = agent.ask(args.ask)
        console.print(Markdown(output))
        return

    run_interactive(agent=agent, console=console)


if __name__ == "__main__":
    main()
