"""
Web Search Tool — agentic web search with multi-engine scraping and LLM synthesis.

Responsibilities
────────────────
1. Rewrite ambiguous queries using conversation history (pronoun resolution).
2. Analyze query intent and enhance the search query via LLM.
3. Execute parallel searches across multiple engines (DuckDuckGo, Bing, Startpage).
4. Deduplicate, rank (LLM-assisted), and synthesize results into a coherent answer.
5. Store structured results in WorkflowState.web_search_results for the API layer.

Design notes
────────────
* All HTTP requests (search engines) use async httpx — never blocks the event loop.
* All LLM calls go through the project's LLMClient with retry logic.
* Search engines are scraped via HTML parsing (BeautifulSoup) — no API keys needed.
* Engines are queried in parallel via asyncio.gather for minimal latency.
* Rate limiting prevents search engine throttling.
* The agentic pipeline (intent → enhance → select engines → search → rank → synthesize)
  is preserved from the original implementation but fully async.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any
from urllib.parse import urlparse, parse_qs, unquote

import httpx
import structlog
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage

from utils.config import *
from utils.apexchat.core.llm import LLMClient, get_web_search_tool_client
from utils.apexchat.schemas.models import ConversationMessage, MessageRole, WorkflowState
from utils.apexchat.tools.general import BaseTool
from utils.apexchat.core.status_stream import emit_status

logger = structlog.get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

_SEARCH_ENGINES: dict[str, dict[str, Any]] = {
    "duckduckgo": {
        "url": "https://html.duckduckgo.com/html/",
        "params_key": "q",
        "timeout": 8,
    },
    "bing": {
        "url": "https://www.bing.com/search",
        "params_key": "q",
        "timeout": 10,
    },
    "startpage": {
        "url": "https://www.startpage.com/sp/search",
        "params_key": "query",
        "timeout": 12,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Search Engine — internal async search + scraping layer
# ══════════════════════════════════════════════════════════════════════════════

class _SearchEngine:
    """
    Async multi-engine web scraper.

    Searches DuckDuckGo, Bing, and Startpage in parallel, parses HTML results
    with BeautifulSoup, deduplicates by URL.
    """

    def __init__(self) -> None:
        self._last_search_time: float = 0.0

    async def search(
        self,
        query: str,
        engines: list[str] | None = None,
        max_results: int = 5,
        timeout: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Execute searches across selected engines in parallel.

        Args:
            query: The (enhanced) search query.
            engines: Engine names to use. Defaults to all enabled.
            max_results: Max results to return after deduplication.
            timeout: Per-engine HTTP timeout in seconds.

        Returns:
            Deduplicated list of result dicts with title, snippet, url, source.
        """
        if not query or not query.strip():
            return []

        engines = engines or list(_SEARCH_ENGINES.keys())

        # Rate limiting
        now = time.monotonic()
        wait = WEB_SEARCH_RATE_LIMIT_SECONDS - (now - self._last_search_time)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_search_time = time.monotonic()

        # Search all engines in parallel
        tasks = [
            self._search_engine(name, query, timeout)
            for name in engines
            if name in _SEARCH_ENGINES
        ]
        engine_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter out exceptions
        all_results: list[dict[str, Any]] = []
        for i, result in enumerate(engine_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Search engine failed",
                    engine=engines[i],
                    error=str(result),
                )
            elif isinstance(result, list):
                all_results.extend(result)

        # Deduplicate by URL
        unique = self._deduplicate(all_results)

        logger.info(
            "Search complete",
            total_raw=len(all_results),
            unique=len(unique),
            engines_used=engines,
        )
        return unique[:max_results]

    # ── Per-engine search methods ─────────────────────────────────────────────

    async def _search_engine(
        self, name: str, query: str, timeout: int,
    ) -> list[dict[str, Any]]:
        """Dispatch to the correct engine parser."""
        cfg = _SEARCH_ENGINES[name]
        engine_timeout = min(timeout, cfg["timeout"])

        async with httpx.AsyncClient(
            headers=_DEFAULT_HEADERS,
            timeout=httpx.Timeout(engine_timeout),
            follow_redirects=True,
        ) as client:
            params = {cfg["params_key"]: query}
            if name == "duckduckgo":
                params["kl"] = "us-en"
            elif name == "bing":
                params["form"] = "QBLH"
                params["mkt"] = "en-US"
            elif name == "startpage":
                params["language"] = "english"

            resp = await client.get(cfg["url"], params=params)
            resp.raise_for_status()
            html = resp.text

        if name == "duckduckgo":
            return self._parse_duckduckgo(html)
        elif name == "bing":
            return self._parse_bing(html)
        elif name == "startpage":
            return self._parse_startpage(html)
        return []

    def _parse_duckduckgo(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for div in soup.find_all("div", class_="result"):
            try:
                title_el = div.find("a", class_=re.compile(r"\bresult__a\b"))
                snippet_el = div.find(
                    ["a", "div"], class_=re.compile(r"\bresult__snippet\b")
                )
                if title_el:
                    title = title_el.get_text(strip=True)
                    url = self._clean_ddg_url(title_el.get("href", ""))
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    if title and url:
                        results.append(
                            {"title": title, "snippet": snippet, "url": url, "source": "DuckDuckGo"}
                        )
            except Exception:
                continue
        return results

    def _parse_bing(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for li in soup.find_all("li", class_="b_algo"):
            try:
                h2 = li.find("h2")
                title_link = h2.find("a") if h2 else None
                caption = li.find("div", class_=re.compile(r"\bb_caption\b"))
                snippet_el = caption.find("p") if caption else li.find("p")
                if title_link:
                    title = title_link.get_text(strip=True)
                    url = title_link.get("href", "")
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    if title and url:
                        results.append(
                            {"title": title, "snippet": snippet, "url": url, "source": "Bing"}
                        )
            except Exception:
                continue
        return results

    def _parse_startpage(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for div in soup.find_all("div", class_="w-gl__result"):
            try:
                h3 = div.find("h3")
                title_link = h3.find("a") if h3 else None
                snippet_el = div.find("p", class_="w-gl__description")
                if title_link:
                    title = title_link.get_text(strip=True)
                    url = title_link.get("href", "")
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    if title and url:
                        results.append(
                            {"title": title, "snippet": snippet, "url": url, "source": "Startpage"}
                        )
            except Exception:
                continue
        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_ddg_url(url: str) -> str:
        """Extract real URL from DuckDuckGo redirect wrapper."""
        try:
            parsed = urlparse(url)
            if parsed.netloc and parsed.scheme:
                return url
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        except Exception:
            pass
        return url

    @staticmethod
    def _deduplicate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate results by URL."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for r in results:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)
        return unique


# ══════════════════════════════════════════════════════════════════════════════
# WebSearchTool — the public BaseTool implementation
# ══════════════════════════════════════════════════════════════════════════════

class WebSearchTool(BaseTool):
    """
    Agentic web search tool with LLM-driven query analysis, enhancement,
    engine selection, ranking, and result synthesis.

    Flow:
        execute()
            -> _rewrite_query_with_context()   pronoun resolution via history
            -> _analyze_intent()               LLM: understand query intent
            -> _enhance_query()                LLM: optimize for search engines
            -> _select_engines()               LLM: pick best engines
            -> _SearchEngine.search()          async parallel scraping
            -> _rank_results()                 LLM-assisted relevance ranking
            -> _synthesize_results()           LLM: coherent answer from sources
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or get_web_search_tool_client()
        self._engine = _SearchEngine()

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Searches the web using multiple search engines and synthesizes "
            "results into a coherent, sourced answer."
        )

    async def execute(self, state: WorkflowState) -> str:
        emit_status("tool_web_search")
        start_time = time.perf_counter()
        raw_query = state.user_message.strip()

        logger.info(
            "WebSearchTool executing",
            session_id=state.session_id,
            message_preview=raw_query[:80],
        )

        # ── Step 1: Rewrite query using conversation context ──────────────────
        search_query = await self._rewrite_query_with_context(
            raw_query, state.conversation_history
        )

        # ── Step 2: Analyze intent ────────────────────────────────────────────
        intent = await self._analyze_intent(search_query)

        # ── Step 3: Enhance query for search engines ──────────────────────────
        enhanced_query = await self._enhance_query(search_query, intent)

        # ── Step 4: Select optimal engines ────────────────────────────────────
        engines = await self._select_engines(intent)

        # ── Step 5: Calculate adaptive parameters ─────────────────────────────
        params = await self._calculate_parameters(intent)
        max_results = params.get("max_results", 5)
        timeout = params.get("timeout", 10)

        # ── Step 6: Execute search ────────────────────────────────────────────
        results = await self._engine.search(
            enhanced_query,
            engines=engines,
            max_results=max_results * 2,  # fetch extra for ranking pass
            timeout=timeout,
        )

        if not results:
            logger.warning("Web search returned 0 results", query=enhanced_query)
            state.web_search_results = {
                "query": search_query,
                "enhanced_query": enhanced_query,
                "results": [],
                "synthesis": None,
            }
            return (
                "I searched the web but couldn't find specific information for your query. "
                "This might be due to network issues or the information not being available online."
            )

        # ── Step 7: Rank results ──────────────────────────────────────────────
        ranked = await self._rank_results(results, search_query, intent)
        final_results = ranked[:max_results]

        # ── Step 8: Synthesize answer ─────────────────────────────────────────
        synthesis = await self._synthesize_results(search_query, final_results)
        answer = synthesis.get("answer", "")

        if not answer:
            answer = (
                f"I found {len(final_results)} results for your query. "
                + (final_results[0].get("snippet", "") if final_results else "")
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "WebSearchTool completed",
            session_id=state.session_id,
            result_count=len(final_results),
            confidence=synthesis.get("confidence", 0),
            elapsed_ms=round(elapsed_ms, 2),
        )

        # ── Step 9: Store results for the API layer ───────────────────────────
        state.web_search_results = {
            "query": search_query,
            "enhanced_query": enhanced_query,
            "results": final_results,
            "synthesis": synthesis,
        }

        return answer

    # ══════════════════════════════════════════════════════════════════════════
    # Query Rewriting (pronoun resolution from conversation history)
    # ══════════════════════════════════════════════════════════════════════════

    async def _rewrite_query_with_context(
        self,
        query: str,
        history: list[ConversationMessage],
    ) -> str:
        """
        Resolve pronouns and references using conversation history.
        e.g. "how much is the population there?" -> "how much is the population in Zahle?"
        """
        reference_words = r"\b(it|there|this|that|he|she|they|them|him|her|here|those|these)\b"
        if not history or not re.search(reference_words, query.lower()):
            return query

        recent = history[-6:]
        history_text = "\n".join(
            f"{msg.role.value}: {msg.content[:200]}" for msg in recent
        )

        prompt = (
            "Given the conversation history, rewrite the user's latest query to be "
            "completely standalone and self-contained. Replace words like \"it\", \"there\", "
            "\"this\", \"that\", \"he\", or \"she\" with the actual entity, location, or subject "
            "they refer to.\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"Latest Query: \"{query}\"\n\n"
            "Respond ONLY with the rewritten query. If already standalone, return it as is."
        )

        try:
            response = await self._llm.ainvoke_with_retry([HumanMessage(content=prompt)])
            rewritten = response.content.strip().strip("\"'") if hasattr(response, "content") else query
            if rewritten and len(rewritten) < 200:
                logger.debug("Query rewritten", original=query, rewritten=rewritten)
                return rewritten
        except Exception as e:
            logger.warning("Query rewrite failed, using original", error=str(e))

        return query

    # ══════════════════════════════════════════════════════════════════════════
    # Agentic LLM Pipeline
    # ══════════════════════════════════════════════════════════════════════════

    async def _llm_invoke(self, prompt: str) -> str:
        """Invoke LLM and return raw content string. Shared by all agentic methods."""
        response = await self._llm.ainvoke_with_retry([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Safely extract first JSON object from LLM response text."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    async def _analyze_intent(self, query: str) -> dict[str, Any]:
        """LLM-driven query intent analysis."""
        prompt = f"""Analyze this search query and extract its intent.

Query: "{query}"

Return JSON:
{{
    "primary_intent": "one of: factual_lookup, price_check, news_update, how_to, comparison, definition, weather, stock_market, crypto, product_search, academic_research, troubleshooting",
    "urgency": "one of: real_time, recent, historical, evergreen",
    "information_need": "brief description of what user wants to know",
    "key_entities": ["main entities/topics"],
    "complexity": "one of: simple, moderate, complex",
    "suggested_sources": ["types of authoritative sources"]
}}

Respond with valid JSON only:"""

        try:
            raw = await self._llm_invoke(prompt)
            parsed = self._extract_json(raw)
            if parsed:
                logger.debug("Intent analyzed", intent=parsed.get("primary_intent"))
                return parsed
        except Exception as e:
            logger.warning("Intent analysis failed", error=str(e))

        return {
            "primary_intent": "factual_lookup",
            "urgency": "evergreen",
            "information_need": query,
            "key_entities": query.split()[:3],
            "complexity": "moderate",
            "suggested_sources": ["general"],
        }

    async def _enhance_query(self, query: str, intent: dict[str, Any]) -> str:
        """LLM-driven query enhancement for better search results."""
        prompt = f"""Enhance this search query for better web search results.

Original Query: "{query}"
Intent: {intent.get('primary_intent')}
Urgency: {intent.get('urgency')}

Guidelines:
- For real_time urgency: add "today", "latest", "current"
- For price queries: add "price", "cost", "current value"
- For news: add "news", "update", "recent"
- For how-to: keep question format, add "guide" or "tutorial"
- Keep query concise (3-8 words optimal)
- Focus on searchable keywords

Return only the enhanced query, nothing else:"""

        try:
            enhanced = (await self._llm_invoke(prompt)).strip().strip("\"'")
            if enhanced and enhanced != query and len(enhanced) < 100:
                logger.debug("Query enhanced", original=query, enhanced=enhanced)
                return enhanced
        except Exception as e:
            logger.warning("Query enhancement failed", error=str(e))

        return query

    async def _select_engines(self, intent: dict[str, Any]) -> list[str]:
        """LLM-driven search engine selection."""
        prompt = f"""Select the best search engines for this query.

Primary Intent: {intent.get('primary_intent')}
Urgency: {intent.get('urgency')}
Suggested Sources: {intent.get('suggested_sources')}

Available:
- duckduckgo: General searches, privacy-focused, decent news coverage
- bing: Commercial/shopping, financial data, up-to-date information
- startpage: Academic/research queries, privacy-focused Google proxy

Return JSON: {{"engines": ["list of 1-3 engines in priority order"]}}

Respond with valid JSON only:"""

        try:
            raw = await self._llm_invoke(prompt)
            parsed = self._extract_json(raw)
            if parsed:
                engines = parsed.get("engines", [])
                valid = [e for e in engines if e in _SEARCH_ENGINES]
                if valid:
                    logger.debug("Engines selected", engines=valid)
                    return valid
        except Exception as e:
            logger.warning("Engine selection failed", error=str(e))

        return list(_SEARCH_ENGINES.keys())

    async def _calculate_parameters(self, intent: dict[str, Any]) -> dict[str, Any]:
        """LLM-driven adaptive parameter calculation."""
        prompt = f"""Determine optimal search parameters.

Primary Intent: {intent.get('primary_intent')}
Complexity: {intent.get('complexity')}
Urgency: {intent.get('urgency')}

Return JSON:
{{
    "max_results": <integer 3-10>,
    "timeout": <integer 5-15>
}}

Simple queries: 3-5 results, 5-8s timeout.
Complex queries: 7-10 results, 10-15s timeout.

Respond with valid JSON only:"""

        try:
            raw = await self._llm_invoke(prompt)
            parsed = self._extract_json(raw)
            if parsed:
                return {
                    "max_results": min(10, max(3, parsed.get("max_results", 5))),
                    "timeout": min(15, max(5, parsed.get("timeout", 10))),
                }
        except Exception as e:
            logger.warning("Parameter calculation failed", error=str(e))

        return {"max_results": 5, "timeout": 10}

    # ══════════════════════════════════════════════════════════════════════════
    # Result Ranking
    # ══════════════════════════════════════════════════════════════════════════

    async def _rank_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        intent: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """LLM-assisted result ranking with keyword fallback."""
        if not results:
            return []

        # Pre-filter large sets with keyword scoring first
        if len(results) > 10:
            results = self._quick_rank(results, query)[:10]

        summaries = [
            {
                "index": i,
                "title": r.get("title", "")[:100],
                "snippet": r.get("snippet", "")[:200],
                "source": r.get("source", ""),
            }
            for i, r in enumerate(results[:10])
        ]

        prompt = f"""Rank these search results by relevance.

Query: "{query}"
Intent: {intent.get('primary_intent')}
Key Entities: {intent.get('key_entities')}

Results:
{json.dumps(summaries, indent=2)}

Return JSON: {{"ranked_indices": [indices in relevance order, e.g. [2, 0, 5, 1]]}}

Consider: direct answer, authoritative sources, recency, completeness, credibility.

Respond with valid JSON only:"""

        try:
            raw = await self._llm_invoke(prompt)
            parsed = self._extract_json(raw)
            if parsed:
                indices = parsed.get("ranked_indices", [])
                if indices:
                    ranked = [results[i] for i in indices if 0 <= i < len(results)]
                    included = set(indices)
                    for i, r in enumerate(results):
                        if i not in included:
                            ranked.append(r)
                    return ranked
        except Exception as e:
            logger.warning("Agentic ranking failed, using keyword fallback", error=str(e))

        return self._quick_rank(results, query)

    @staticmethod
    def _quick_rank(results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        """Fast keyword-based ranking as fallback."""
        query_words = set(query.lower().split())

        def score(r: dict[str, Any]) -> int:
            title = r.get("title", "").lower()
            snippet = r.get("snippet", "").lower()
            s = sum(3 for w in query_words if w in title)
            s += sum(1 for w in query_words if w in snippet)
            if query.lower() in title:
                s += 5
            return s

        return sorted(results, key=score, reverse=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Result Synthesis
    # ══════════════════════════════════════════════════════════════════════════

    async def _synthesize_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        max_sentences: int = 5,
    ) -> dict[str, Any]:
        """LLM-driven synthesis of search results into a coherent answer."""
        if not results:
            return {"answer": "", "confidence": 0.0, "citations": []}

        context_parts = []
        for i, r in enumerate(results[:8]):
            snippet = r.get("snippet", "").strip()
            title = r.get("title", "").strip()
            domain = self._extract_domain(r.get("url", ""))
            text = snippet or title
            if text:
                context_parts.append(f"Source {i + 1} ({domain}): {text}")

        context = "\n\n".join(context_parts)

        prompt = f"""Synthesize a comprehensive answer from these search results.

Query: "{query}"

Search Results:
{context}

Instructions:
1. Synthesize information from multiple sources into a coherent answer
2. Focus on directly answering the query
3. Include specific data (numbers, dates, names) when available
4. Write {max_sentences} clear, informative sentences
5. If results conflict, mention both perspectives

Return JSON:
{{
    "answer": "synthesized answer in {max_sentences} sentences",
    "confidence": 0.0 to 1.0,
    "key_sources": ["2-3 most important source domains"]
}}

Respond with valid JSON only:"""

        try:
            raw = await self._llm_invoke(prompt)
            parsed = self._extract_json(raw)
            if parsed:
                answer = parsed.get("answer", "")
                confidence = parsed.get("confidence", 0.5)
                key_sources = parsed.get("key_sources", [])

                # Build citations from key sources
                citations = []
                for domain in key_sources:
                    for r in results:
                        if domain.lower() in r.get("url", "").lower():
                            citations.append({
                                "domain": domain,
                                "link": r.get("url", ""),
                                "title": r.get("title", ""),
                            })
                            break

                # Append source attribution
                if citations:
                    source_names = [c["domain"] for c in citations]
                    answer += f"\n\nSources: {', '.join(source_names)}"

                return {
                    "answer": answer,
                    "confidence": confidence,
                    "citations": citations,
                }
        except Exception as e:
            logger.warning("LLM synthesis failed, using basic fallback", error=str(e))

        return self._basic_synthesis(results, max_sentences)

    @staticmethod
    def _basic_synthesis(
        results: list[dict[str, Any]], max_sentences: int,
    ) -> dict[str, Any]:
        """Non-LLM fallback synthesis from raw snippets."""
        snippets = []
        citations = []
        for r in results[:max_sentences]:
            snippet = r.get("snippet", "").strip()
            if snippet and len(snippet) > 30:
                snippets.append(snippet)
                domain = WebSearchTool._extract_domain(r.get("url", ""))
                if domain:
                    citations.append({
                        "domain": domain,
                        "link": r.get("url", ""),
                        "title": r.get("title", ""),
                    })

        answer = " ".join(snippets[:max_sentences])
        if citations:
            unique = list({c["domain"]: c for c in citations}.values())[:3]
            answer += f"\n\nSources: {', '.join(c['domain'] for c in unique)}"

        return {"answer": answer, "confidence": 0.6, "citations": citations}

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract clean domain name from URL."""
        try:
            domain = urlparse(url).netloc
            return domain[4:] if domain.startswith("www.") else domain
        except Exception:
            return "Unknown"