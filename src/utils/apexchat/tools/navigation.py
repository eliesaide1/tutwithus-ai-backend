"""
Navigation Tool — resolves a user's screen navigation intent to a screen route.

Screen mappings are hardcoded — no database lookup required.
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel

from utils.apexchat.core.llm import LLMClient, get_navigation_tool_client
from utils.apexchat.schemas.models import WorkflowState
from utils.apexchat.tools.general import BaseTool
from utils.apexchat.core.status_stream import emit_status

logger = structlog.get_logger(__name__)


# ── Hardcoded Screen Mappings ─────────────────────────────────────────────────
# Maps semantic aliases -> canonical screen route sent to the front-end.

HARDCODED_SCREENS: dict[str, str] = {
    # Booking
    "booking":                  "booking",
    "book":                     "booking",
    "book a session":           "booking",
    "book session":             "booking",
    "schedule session":         "booking",

    # Bundles (default tab)
    "bundles":                  "bundles",
    "bundle":                   "bundles",
    "packages":                 "bundles",

    # Bundles — credits tab
    "credits":                  "bundles?tab=credits",
    "bundles?tab=credits":      "bundles?tab=credits",
    "bundles credits":          "bundles?tab=credits",
    "credit":                   "bundles?tab=credits",
    "buy credits":              "bundles?tab=credits",
    "purchase credits":         "bundles?tab=credits",

    # Bundles — bundles tab
    "bundles?tab=bundles":      "bundles?tab=bundles",
    "bundles tab":              "bundles?tab=bundles",
    "bundle packages":          "bundles?tab=bundles",

    # Courses
    "courses":                  "courses",
    "course":                   "courses",
    "classes":                  "courses",
    "lessons":                  "courses",

    # Tutors
    "tutors":                   "tutors",
    "tutor":                    "tutors",
    "teachers":                 "tutors",
    "instructors":              "tutors",
    "find a tutor":             "tutors",

    # About
    "about":                    "about",
    "about us":                 "about",
    "company":                  "about",
    "who are you":              "about",

    # Checkout
    "checkout":                 "checkout",
    "check out":                "checkout",
    "payment":                  "checkout",
    "pay":                      "checkout",
    "purchase":                 "checkout",

    # FAQ
    "faq":                      "faq",
    "faqs":                     "faq",
    "frequently asked":         "faq",
    "questions":                "faq",
    "help":                     "faq",

    # Blogs
    "blogs":                    "blogs",
    "blog":                     "blogs",
    "articles":                 "blogs",
    "posts":                    "blogs",
    "news":                     "blogs",

    # Profile (root)
    "profile":                  "profile",
    "my profile":               "profile",
    "account":                  "profile",
    "my account":               "profile",

    # Profile — schedule
    "profile/schedule":         "profile/schedule",
    "profile schedule":         "profile/schedule",
    "my schedule":              "profile/schedule",
    "schedule":                 "profile/schedule",
    "availability":             "profile/schedule",
    "my availability":          "profile/schedule",

    # Profile — about me
    "profile/aboutme":          "profile/aboutMe",
    "profile aboutme":          "profile/aboutMe",
    "about me":                 "profile/aboutMe",
    "my info":                  "profile/aboutMe",
    "my details":               "profile/aboutMe",
    "personal info":            "profile/aboutMe",
}


# ── Structured Output Schema ──────────────────────────────────────────────────

class ScreenNameSchema(BaseModel):
    """Structured output for screen name extraction."""
    screen_name: str | None = None


# ── NavigationTool ────────────────────────────────────────────────────────────

class NavigationTool(BaseTool):
    """
    Handles screen navigation requests using hardcoded screen mappings.

    Flow:
        execute()
            -> _extract_screen_name()     LLM structured output
            -> _resolve_to_object_id()    exact match -> fuzzy fallback
            -> _get_suggestions()         only on miss
    """

    _EXTRACT_PROMPT: str = """You are a navigation assistant for an online tutoring platform.

Your job is to identify which screen the user wants to navigate to.

Available screens and what they are for:
- booking          → book or schedule a tutoring session
- bundles          → view available bundles/packages (default view)
- bundles?tab=credits  → buy or manage credits
- bundles?tab=bundles  → view bundle packages specifically
- courses          → browse available courses, classes, or lessons
- tutors           → find or browse tutors/teachers/instructors
- about            → learn about the company / about us page
- checkout         → payment or checkout page to complete a purchase
- faq              → frequently asked questions / help
- blogs            → read blog posts, articles, or news
- profile          → view or edit the user's profile / account
- profile/schedule → manage the user's schedule or availability
- profile/aboutMe  → edit the user's personal information / about me section

Rules:
- Map the user's intent to the closest screen name from the list above (use the exact key, e.g. "bundles?tab=credits").
- If the user asks about credits, buying credits, or purchasing credits → return "bundles?tab=credits".
- If the user asks about their schedule or availability → return "profile/schedule".
- If the user asks about their personal info or about me → return "profile/aboutMe".
- If no screen matches, return null.

User query:
{query}"""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or get_navigation_tool_client()
        self._screen_mappings: dict[str, str] = HARDCODED_SCREENS

    # ── BaseTool interface ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "navigation"

    @property
    def description(self) -> str:
        return (
            "Resolves a user's screen navigation intent to a system screen identifier "
            "and populates nav_data so the front-end can trigger the screen transition."
        )

    async def execute(self, state: WorkflowState) -> str:
        emit_status("tool_navigation")
        start_time = time.perf_counter()
        query = state.user_message.strip()

        logger.info(
            "NavigationTool executing",
            session_id=state.session_id,
            message_preview=query[:80],
        )

        screen_name = await self._extract_screen_name(query)
        print(f"Extracted screen name: {screen_name}")

        if not screen_name:
            logger.warning(
                "NavigationTool: no screen name extracted",
                session_id=state.session_id,
            )
            return (
                "I couldn't identify which screen you want to navigate to. "
                "Could you be more specific? For example: *\"open the transactions screen\"*."
            )

        object_id = await self._resolve_to_object_id(screen_name)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if object_id:
            state.nav_data = {
                "screen_name": screen_name,
                "object_id": object_id,
            }
            logger.info(
                "NavigationTool resolved",
                session_id=state.session_id,
                screen_name=screen_name,
                object_id=object_id,
                elapsed_ms=round(elapsed_ms, 2),
            )
            return f"Opening **{screen_name}** screen..."

        suggestions = self._get_suggestions(screen_name)
        logger.info(
            "NavigationTool: screen not found",
            session_id=state.session_id,
            screen_name=screen_name,
            suggestion_count=len(suggestions),
            elapsed_ms=round(elapsed_ms, 2),
        )

        response = f"I couldn't find a screen named **\"{screen_name}\"**."
        if suggestions:
            formatted = "\n".join(f"- {s}" for s in suggestions)
            response += f"\n\nDid you mean one of these?\n{formatted}"
        else:
            response += (
                "\n\nNo similar screens were found. "
                "Please check the screen name and try again."
            )
        return response

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _extract_screen_name(self, query: str) -> str | None:
        structured_llm = self._llm.with_structured_output(ScreenNameSchema)
        prompt = self._EXTRACT_PROMPT.format(query=query)

        try:
            parsed: ScreenNameSchema = await structured_llm.ainvoke(prompt)

            if parsed.screen_name and parsed.screen_name.lower() not in ("null", "none", ""):
                extracted = parsed.screen_name.strip().lower()
                logger.debug("NavigationTool: screen name extracted", screen_name=extracted)
                return extracted

            return None

        except Exception as e:
            logger.error(
                "NavigationTool: screen name extraction failed",
                error=str(e),
                exc_info=True,
            )
            return None

    async def _resolve_to_object_id(self, screen_name: str) -> str | None:
        if not screen_name:
            return None

        name_lower = screen_name.strip().lower()

        if name_lower in self._screen_mappings:
            return self._screen_mappings[name_lower]

        return self._fuzzy_match(name_lower)

    def _fuzzy_match(self, screen_name: str) -> str | None:
        for key, value in self._screen_mappings.items():
            if screen_name in key or key in screen_name:
                logger.debug(
                    "NavigationTool: substring match",
                    query=screen_name,
                    matched_key=key,
                )
                return value

        query_words = set(screen_name.split())
        best_value: str | None = None
        best_score: int = 0

        for key, value in self._screen_mappings.items():
            overlap = len(query_words & set(key.split()))
            if overlap > best_score and overlap >= len(query_words) * 0.5:
                best_score = overlap
                best_value = value

        if best_value:
            logger.debug(
                "NavigationTool: word-overlap match",
                query=screen_name,
                score=best_score,
            )

        return best_value

    def _get_suggestions(self, screen_name: str) -> list[str]:
        name_lower = screen_name.lower()
        query_words = name_lower.split()
        suggestions: set[str] = set()

        for key in self._screen_mappings:
            if name_lower in key or any(word in key for word in query_words):
                suggestions.add(key)

        return sorted(suggestions)[:5]

