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
    # Auth — sign in (public, only useful when signed out)
    "login":                    "login",
    "log in":                   "login",
    "log into my account":      "login",
    "sign in":                  "login",
    "signin":                   "login",

    # Auth — sign up / register (public, only useful when signed out)
    "sign up":                  "sign-up",
    "signup":                   "sign-up",
    "sign-up":                  "sign-up",
    "register":                 "sign-up",
    "registration":             "sign-up",
    "create account":          "sign-up",
    "create an account":       "sign-up",
    "make an account":         "sign-up",
    "new account":             "sign-up",
    "join":                     "sign-up",

    # Auth — forgot / reset password (public)
    "forgot password":          "forgot-password",
    "forgot my password":       "forgot-password",
    "reset password":           "forgot-password",
    "forgot-password":          "forgot-password",

    # Booking
    "booking":                  "booking",
    "book":                     "booking",
    "book a session":           "booking",
    "book session":             "booking",
    "schedule session":         "booking",

    # Bundles (public by default; admin/bundles when in the admin panel)
    "bundles":                  "bundles",
    "bundle":                   "bundles",
    "packages":                 "bundles",

    # Credits — public buy-credits tab by default; admin/credits when in admin.
    "credits":                  "bundles?tab=credits",
    "credit":                   "bundles?tab=credits",
    "buy credits":              "bundles?tab=credits",
    "purchase credits":         "bundles?tab=credits",
    "bundles?tab=credits":      "bundles?tab=credits",

    # Courses
    "courses":                  "courses",
    "course":                   "courses",
    "classes":                  "courses",
    "lessons":                  "courses",

    # Tutors (public listing by default; remapped to admin/tutors when the
    # user is currently in the admin panel — see _resolve_with_context).
    "tutors":                   "tutors",
    "tutor":                    "tutors",
    "teachers":                 "tutors",
    "instructors":              "tutors",
    "find a tutor":             "tutors",
    "tutors screen":            "tutors",

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

    # FAQ (public by default; admin/faq when in the admin panel)
    "faq":                      "faq",
    "faqs":                     "faq",
    "frequently asked":         "faq",
    "questions":                "faq",
    "help":                     "faq",

    # Blogs (public by default; admin/blogs when in the admin panel)
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

    # Schedule / availability — the user's own schedule. (Not context-remapped:
    # bare and "my …" phrasings share this route, so we can't safely flip it;
    # use the explicit "admin schedule" alias below for admin availability.)
    "schedule":                 "profile/schedule",
    "availability":             "profile/schedule",
    "profile/schedule":         "profile/schedule",
    "profile schedule":         "profile/schedule",
    "my schedule":              "profile/schedule",
    "my availability":          "profile/schedule",

    # Profile — about me
    "profile/aboutme":          "profile/aboutMe",
    "profile aboutme":          "profile/aboutMe",
    "about me":                 "profile/aboutMe",
    "my info":                  "profile/aboutMe",
    "my details":               "profile/aboutMe",
    "personal info":            "profile/aboutMe",

    # ── Admin area (all under /admin, require admin access) ───────────────────
    # Admin dashboard root
    "admin":                    "admin",
    "admin dashboard":          "admin",
    "admin panel":              "admin",
    "dashboard":                "admin",

    # Promo codes (admin-only — no public equivalent, bare keywords are safe)
    "promo":                    "admin/promo",
    "promos":                   "admin/promo",
    "promo code":               "admin/promo",
    "promo codes":              "admin/promo",
    "promocodes":               "admin/promo",
    "promotion":                "admin/promo",
    "promotions":               "admin/promo",
    "discount code":            "admin/promo",
    "discount codes":           "admin/promo",
    "coupon":                   "admin/promo",
    "coupons":                  "admin/promo",
    "voucher":                  "admin/promo",
    "vouchers":                 "admin/promo",
    "admin promo":              "admin/promo",
    "manage promo":             "admin/promo",

    # Statistics (admin-only)
    "statistics":               "admin/statistics",
    "stats":                    "admin/statistics",
    "analytics":                "admin/statistics",
    "admin statistics":         "admin/statistics",
    "manage statistics":        "admin/statistics",

    # Subjects (admin-only — no public subjects page)
    "subjects":                 "admin/subjects",
    "subject":                  "admin/subjects",
    "admin subjects":           "admin/subjects",
    "manage subjects":          "admin/subjects",

    # Levels (admin-only)
    "levels":                   "admin/levels",
    "level":                    "admin/levels",
    "grade levels":             "admin/levels",
    "admin levels":             "admin/levels",
    "manage levels":            "admin/levels",

    # Benefits (admin-only)
    "benefits":                 "admin/benefits",
    "benefit":                  "admin/benefits",
    "admin benefits":           "admin/benefits",
    "manage benefits":          "admin/benefits",

    # Referrals (note: route folder is misspelled "refferal")
    "referrals":                "admin/refferal",
    "referral":                 "admin/refferal",
    "referral program":         "admin/refferal",
    "admin referrals":          "admin/refferal",
    "manage referrals":         "admin/refferal",

    # Hero section (admin-only)
    "hero":                     "admin/hero",
    "hero section":             "admin/hero",
    "homepage hero":            "admin/hero",
    "admin hero":               "admin/hero",
    "manage hero":              "admin/hero",

    # Extra "admin …" / "manage …" phrasings (the bare nouns above already
    # resolve to these same admin routes).
    "admin tutors":             "admin/tutors",
    "manage tutors":            "admin/tutors",
    "tutors management":        "admin/tutors",

    "admin bundles":            "admin/bundles",
    "manage bundles":           "admin/bundles",
    "bundles management":       "admin/bundles",

    "admin blogs":              "admin/blogs",
    "manage blogs":             "admin/blogs",
    "blogs management":         "admin/blogs",
    "manage articles":          "admin/blogs",

    "admin faq":                "admin/faq",
    "manage faq":               "admin/faq",
    "faq management":           "admin/faq",

    "admin credits":            "admin/credits",
    "manage credits":           "admin/credits",
    "credits management":       "admin/credits",

    "admin schedule":           "admin/schedule",
    "manage schedule":          "admin/schedule",
    "admin availability":       "admin/schedule",

    # Self-mapped canonical admin routes so an LLM that returns the route key
    # (e.g. "admin/promo") resolves exactly instead of fuzzy-matching "admin".
    "admin/promo":              "admin/promo",
    "admin/statistics":         "admin/statistics",
    "admin/subjects":           "admin/subjects",
    "admin/levels":             "admin/levels",
    "admin/benefits":           "admin/benefits",
    "admin/refferal":           "admin/refferal",
    "admin/hero":               "admin/hero",
    "admin/tutors":             "admin/tutors",
    "admin/bundles":            "admin/bundles",
    "admin/blogs":              "admin/blogs",
    "admin/faq":                "admin/faq",
    "admin/credits":            "admin/credits",
    "admin/schedule":           "admin/schedule",
}


# ── Structured Output Schema ──────────────────────────────────────────────────

class ScreenNameSchema(BaseModel):
    """Structured output for screen name extraction."""
    screen_name: str | None = None


# ── Auth gating ───────────────────────────────────────────────────────────────
# Screens that may only be opened by a signed-in user. A guest asking for any of
# these is routed to the sign-in page instead. Matched against the resolved
# route's base (query string stripped) so "profile/schedule" etc. are covered.
PROTECTED_PREFIXES: tuple[str, ...] = ("booking", "checkout", "profile", "admin")

# Auth screens — only useful when signed OUT. A signed-in user asking for one of
# these is told they're already signed in rather than being bounced by the
# front-end middleware.
AUTH_SCREENS: frozenset[str] = frozenset({"login", "sign-up", "forgot-password"})

# Canonical route a guest is sent to when they request a protected screen.
LOGIN_SCREEN: str = "login"

# Pages that exist in BOTH the public site and the admin dashboard. When the
# user is currently inside the admin panel, a request that resolves to the
# public route is remapped to its admin counterpart (e.g. "go to bundles" on
# /en/admin/* → admin/bundles, but on the public site → bundles).
PUBLIC_TO_ADMIN: dict[str, str] = {
    "tutors":               "admin/tutors",
    "bundles":              "admin/bundles",
    "faq":                  "admin/faq",
    "blogs":                "admin/blogs",
    "bundles?tab=credits":  "admin/credits",
}


def _in_admin_context(state: WorkflowState) -> bool:
    """True when the front-end says the user is currently in the admin panel."""
    path = (state.current_path or "").lower()
    segments = [seg for seg in path.split("/") if seg]
    return "admin" in segments


def _base_route(object_id: str) -> str:
    """Return the route without any query string, lower-cased."""
    return object_id.split("?", 1)[0].strip().lower()


def _is_auth_screen(object_id: str) -> bool:
    return _base_route(object_id) in AUTH_SCREENS


def _requires_auth(object_id: str) -> bool:
    base = _base_route(object_id)
    return any(base == p or base.startswith(f"{p}/") for p in PROTECTED_PREFIXES)


def _is_guest(state: WorkflowState) -> bool:
    """Mirror the guest check used by the booking/rescheduling tools."""
    return not state.user_id or "guest" in state.user_id


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

Just identify the page the user means and return its base screen key. Do NOT
add the "admin/" prefix for pages that exist on both the public site and the
admin dashboard (tutors, bundles, faq, blogs, credits) — the system adds the
admin prefix automatically based on where the user currently is.

Available screens and what they are for:
- login            → sign in / log in to an existing account
- sign-up          → create a new account / register / join
- forgot-password  → reset or recover a forgotten password
- booking          → book or schedule a tutoring session
- bundles          → bundles / packages
- bundles?tab=credits  → buy / purchase credits
- courses          → browse available courses, classes, or lessons
- tutors           → tutors / teachers / instructors
- about            → learn about the company / about us page
- checkout         → payment or checkout page to complete a purchase
- faq              → FAQ / help / questions
- blogs            → blogs / articles / posts / news
- profile          → view or edit the user's own profile / account
- profile/schedule → the user's own schedule / availability ("my schedule")
- profile/aboutMe  → edit the user's personal information / about me section

Admin-only screens (no public version — always return the "admin/..." key):
- admin             → the admin dashboard home / panel
- admin/promo       → promo codes / discount codes / coupons
- admin/statistics  → statistics / analytics
- admin/subjects    → subjects
- admin/levels      → grade levels
- admin/benefits    → benefits
- admin/refferal    → the referral program (note the spelling)
- admin/hero        → the homepage hero section

Rules:
- For shared pages, return the BASE key (e.g. "tutors", "bundles", "faq", "blogs") — never "admin/tutors". The system decides public vs admin from the user's current location.
- Admin-only screens (promo codes, statistics, subjects, grade levels, benefits, referrals, hero) → return the "admin/..." key even if the user does not say "admin".
- "admin", "admin dashboard", or "admin panel" → return "admin".
- Buying / purchasing credits → return "bundles?tab=credits".
- If the user wants to sign in / log in → return "login".
- If the user wants to sign up / register / create an account / join → return "sign-up".
- If the user wants to reset or recover a forgotten password → return "forgot-password".
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
            is_guest = _is_guest(state)

            # Context-aware remap: a page name that exists in both areas resolves
            # to the admin route when the user is currently inside the admin
            # panel, and to the public route otherwise.
            if _in_admin_context(state) and object_id in PUBLIC_TO_ADMIN:
                admin_route = PUBLIC_TO_ADMIN[object_id]
                logger.info(
                    "NavigationTool: admin-context remap",
                    session_id=state.session_id,
                    public_route=object_id,
                    admin_route=admin_route,
                    current_path=state.current_path,
                )
                object_id = admin_route
                screen_name = admin_route

            # Signed-in user asking for an auth page (login/sign-up/…): there's
            # nothing for them there — the front-end would just bounce them back.
            if _is_auth_screen(object_id) and not is_guest:
                logger.info(
                    "NavigationTool: auth screen requested while signed in",
                    session_id=state.session_id,
                    screen_name=screen_name,
                )
                return (
                    "You're already signed in 🎉 — no need to visit the "
                    f"**{screen_name}** page. Is there anything else I can help "
                    "you with?"
                )

            # Guest asking for a protected page: send them to sign-in instead of
            # navigating somewhere the front-end won't let them stay.
            if _requires_auth(object_id) and is_guest:
                state.nav_data = {
                    "screen_name": LOGIN_SCREEN,
                    "object_id": LOGIN_SCREEN,
                }
                logger.info(
                    "NavigationTool: guest redirected to sign-in",
                    session_id=state.session_id,
                    requested_screen=screen_name,
                    requested_object_id=object_id,
                    elapsed_ms=round(elapsed_ms, 2),
                )
                return (
                    f"The **{screen_name}** page requires you to be signed in. "
                    "Taking you to the sign-in page now — once you're signed in, "
                    "just ask me again and I'll take you there."
                )

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

