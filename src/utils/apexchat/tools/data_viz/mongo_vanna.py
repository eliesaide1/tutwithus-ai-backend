"""
MongoVanna — a Vanna-AI-style engine adapted for MongoDB.

Two-mode design
---------------
1. ``ask(question)``  — NL question → one or more pipelines + a polished
                        multi-chart dashboard HTML template.  Does NOT execute
                        the pipelines; the frontend runs them on its refresh
                        schedule.
2. ``run(collection, …)`` — execute a previously-generated pipeline and return
                            JSON-safe rows.  Backs ``POST /api/v1/data_viz/run``.
                            Optionally enriches opaque ID fields with the
                            referenced collection's label field (``name`` /
                            ``title`` / etc.) when an ``enrichment`` block is
                            supplied.

Returned bundle from ``ask`` (dict)::

    {
        "success":        True,
        "question":       "...",
        "dashboard_id":   "viz-...",
        "dashboard_title":"Top-level title",
        "dashboard_description":"Top-level description",
        "theme":          "light" | "dark",
        "explanation":    "...",
        "charts": [
            {
                "id":            "viz-...-c1",
                "collection":    "users",
                "pipeline":      [...],
                "output_columns":["role", "count"],
                "chart_type":    "bar"|"line"|"pie"|"doughnut"|"scatter"
                                  |"area"|"radar"|"polarArea"|"table",
                "chart_x":       "role",
                "chart_y":       "count",
                "chart_title":   "Users by role",
                "chart_description":"What this chart shows and why it matters",
                "chart_subtitle":"...",
                "x_axis_label":  "Role",
                "y_axis_label":  "Count",
                "colors":        ["#818cf8", "#06b6d4", ...],
                "stacked":       False,
                "enrichment": [
                    {"column": "tutor_id",
                     "lookup_collection": "users",
                     "label_field": "name"}
                ],
                "size":          "sm"|"md"|"lg"|"full",
                "sort_by":       "x"|"y"|null,
                "sort_dir":      "asc"|"desc",
                "limit":         200
            },
            ...
        ],
        "dashboard_html":         "<div ...>__CHART_DATA__...</div>",
        "chart_data_placeholder": "__CHART_DATA__",
        # Legacy single-chart fields — populated from charts[0] for backward
        # compatibility with callers that don't know about `charts`.
        "collection":     "...",
        "pipeline":       [...],
        "chart_type":     "...",
        "chart_x":        "...",
        "chart_y":        "...",
        "chart_title":    "...",
        "chart_html":     "...",
        "training_examples_used": int
    }

Frontend refresh loop (per chart)::

    1. Read each chart's `pipeline`, `collection`, and optional `enrichment`.
    2. POST them to `/api/v1/data_viz/run` to get fresh, enriched rows.
    3. Replace the literal token `__CHART_DATA__` inside `dashboard_html` with
       a JSON object keyed by chart id (``{"viz-...-c1": rows, ...}``), OR
       call ``window.MongoVannaDashboards["<dashboard_id>"].refresh(allRows)``.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import math
import re
import uuid
from datetime import datetime, date
from typing import Any

import structlog
from bson import ObjectId
from langchain_core.messages import HumanMessage, SystemMessage
from pymongo.database import Database

from utils.apexchat.core.llm import LLMClient, get_dashboard_tool_client
from utils.apexchat.tools.data_viz.training import (
    SCHEMA_DOCUMENTATION,
    SEED_EXAMPLES,
    introspect_schema,
    schema_to_prompt_text,
)

logger = structlog.get_logger(__name__)

# ── Constants used by callers / template ─────────────────────────────────────

CHART_DATA_PLACEHOLDER = "__CHART_DATA__"

# Curated palettes — referenced by chart spec as palette names OR explicit
# hex arrays.  Picked to look professional in both light & dark themes.
PALETTES: dict[str, list[str]] = {
    "indigo":   ["#818cf8", "#6366f1", "#a5b4fc", "#c7d2fe", "#e0e7ff", "#4f46e5"],
    "ocean":    ["#06b6d4", "#0ea5e9", "#38bdf8", "#7dd3fc", "#3b82f6", "#818cf8"],
    "emerald":  ["#34d399", "#10b981", "#6ee7b7", "#a7f3d0", "#22c55e", "#84cc16"],
    "sunset":   ["#fb923c", "#f97316", "#f43f5e", "#ec4899", "#a855f7", "#818cf8"],
    "graphite": ["#475569", "#64748b", "#94a3b8", "#cbd5e1", "#e2e8f0", "#f1f5f9"],
    "default":  ["#818cf8", "#06b6d4", "#34d399", "#fb923c", "#f472b6", "#facc15",
                 "#a78bfa", "#0ea5e9"],
}

_VALID_CHART_TYPES = {
    "bar", "line", "pie", "doughnut", "scatter",
    "area", "radar", "polarArea", "table",
}


# ── JSON-safe encoder (handles ObjectId, datetime, NaN) ──────────────────────

def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


# ── Pipeline JSON cleanup (LLMs love wrapping in fences) ─────────────────────

_FENCE_RE = re.compile(r"```(?:json|javascript|js)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


def _resolve_palette(spec: Any) -> list[str]:
    """Return a concrete list of hex colors from a name OR a list of hex codes."""
    if isinstance(spec, list) and spec:
        cleaned = [c for c in spec if isinstance(c, str) and c.startswith("#")]
        if cleaned:
            return cleaned
    if isinstance(spec, str) and spec in PALETTES:
        return PALETTES[spec]
    return PALETTES["default"]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
    except ValueError:
        return hex_color
    return f"rgba({r},{g},{b},{alpha})"


def _clean_metadata_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _humanise_identifier(value: Any) -> str:
    text = _clean_metadata_text(str(value or ""))
    if not text:
        return ""
    text = text.rsplit(".", 1)[-1]
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(word.upper() if word.lower() == "id" else word.capitalize() for word in text.split())


def _summarise_question_intent(question: str) -> str:
    text = _clean_metadata_text(question)
    if not text:
        return ""
    text = re.sub(r"[\"'`]+", "", text)
    text = re.sub(
        r"^(please\s+)?(can|could|would)\s+you\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(please\s+)?(show|give|create|build|generate|make|display|plot|visualize|render)\s+(me\s+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(a|an|the)?\s*(dashboard|chart|charts|graph|graphs|visualization|visualizations|report)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(of|for|about|with)\b\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .?!:;,-")
    return text


def _title_case_phrase(text: str, max_words: int = 10) -> str:
    words = _clean_metadata_text(text).split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(word.upper() if word.lower() == "id" else word.capitalize() for word in words)


def _join_human_list(items: list[str]) -> str:
    clean = [item for item in items if item]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def _chart_type_label(chart_type: str) -> str:
    labels = {
        "polarArea": "polar area chart",
        "doughnut": "doughnut chart",
        "table": "table",
    }
    return labels.get(chart_type, f"{_humanise_identifier(chart_type).lower()} chart")


# ── Single-shot prompt: pipelines + chart specs + explanation ────────────────

_PLAN_SYSTEM_PROMPT = """You are MongoVanna, an expert at translating natural-language analytics
questions into MongoDB aggregation pipelines AND choosing the best chart for the result.

You will be given:
  • A user question (which may explicitly request multiple charts and per-chart customisation
    such as colours, axis labels, chart types).
  • The current database schema (collections, field paths, inferred types, reference hints).
  • A short list of training examples in the "house style" (use $lookup to resolve IDs).

You MUST return STRICT JSON ONLY (no prose, no markdown fences) of this exact form:

{
  "dashboard_title": "<short overall title for the dashboard>",
  "dashboard_description": "<one or two concise sentences describing the dashboard's purpose>",
  "explanation":     "<one or two sentences describing what the user will see>",
  "theme":           "light" | "dark",
  "charts": [
    {
      "collection":     "<exact collection name from the schema>",
      "pipeline":       [ <BSON-style aggregation stages> ],
      "output_columns": ["<col1>", "<col2>", ...],
      "chart_type":     "bar" | "line" | "pie" | "doughnut" | "scatter"
                          | "area" | "radar" | "polarArea" | "table",
      "chart_x":        "<column for x axis / labels>",
      "chart_y":        "<column for y axis / values>",
      "chart_title":    "<short chart title>",
      "chart_description": "<one or two concise sentences explaining this chart>",
      "chart_subtitle": "<optional one-line subtitle; use only if distinct from chart_description>",
      "x_axis_label":   "<axis label, optional>",
      "y_axis_label":   "<axis label, optional>",
      "colors":         "indigo" | "ocean" | "emerald" | "sunset" | "graphite"
                          | ["#hex", ...],
      "stacked":        false,
      "enrichment": [
        { "column": "tutor_id",
          "lookup_collection": "users",
          "label_field": "name" }
      ],
      "size":   "sm" | "md" | "lg" | "full",
      "sort_by":"x" | "y" | null,
      "sort_dir":"asc" | "desc",
      "limit":   200
    }
    /* ... more charts as needed ... */
  ]
}

# House rules — non-negotiable

1. **Resolve IDs to names.**  If a chart's x/y references a foreign-key field
   (anything matching `_id`, `*_id`, `*Id`, `*Ref`, `*_uuid`), the pipeline
   MUST `$lookup` the referenced collection and `$project` the human-readable
   label.  NEVER plot raw ObjectIds on a user-facing axis.  Use the schema's
   reference hints to find the right collection and label field.

   • **Multilingual labels.**  When the referenced collection's label field
     is a dict like `{en: "...", ar: "..."}` (the schema marks these as type
     `object`, with dotted children like `name.en`), project via
     `$ifNull: ["$x.name.en", "$x.name.ar", "Unknown"]`.  NEVER project the
     raw `name` object — it renders as "[object Object]".
   • **People names.**  The `users` collection has no single `name` field —
     names are split across `firstName` and `lastName`.  When projecting a
     user label, ALWAYS concatenate them:
     `{ "$trim": { "input": { "$concat": [
        {"$ifNull": ["$user.firstName", ""]}, " ",
        {"$ifNull": ["$user.lastName", ""]}
     ] } } }`.
   • **`$unwind` after `$lookup`** MUST set
     `preserveNullAndEmptyArrays: true` so rows whose foreign key points to
     a deleted/missing record are kept (with a null label) rather than
     silently dropped.
2. Pick a collection that EXISTS in the schema.  Never invent one.
3. `output_columns` MUST list the field names that the FINAL stage emits
   (after `$project` / `$group` renames).  `chart_x` and `chart_y` MUST be
   members of `output_columns`.
4. Always cap each pipeline with `$limit` (default 200) unless the user asks
   otherwise — even after `$sort`.
5. Use `$group` for any "by", "per", "count", "sum", "average", "distribution"
   question.  For time-series, group by date with `$dateTrunc` or
   `$dateToString` and sort ascending → use `chart_type="line"` or "area".
6. Categorical breakdowns:
     • ≤ 6 categories  → `pie` or `doughnut`
     • > 6 categories → `bar`
   Two numeric columns where the user wants correlation → `scatter`.
7. **Multi-chart requests.**  When the user asks for several views in one
   message, OR uses words like "and", "also", "compare", "alongside", emit
   multiple chart objects in `charts`.  Each chart is independent.
8. Honour explicit user customisation: chart type, colours, and axis labels —
   copy them into the chart spec verbatim when given.
9. **Dashboard and chart metadata.**  Extract any dashboard title,
   dashboard description, chart title, and chart description the user provides.
   If any are missing, generate concise, professional labels from the user's
   intent, chart fields, grouping, aggregation, collection, and time filters.
   Every dashboard and every chart MUST have a non-empty title and description.
10. Pipelines may use only safe stages: `$match`, `$group`, `$project`,
    `$sort`, `$limit`, `$bucket`, `$bucketAuto`, `$unwind`, `$lookup`,
    `$addFields`, `$count`, `$facet`.  NEVER use `$$ROOT`, `$function`,
    `$accumulator`, `$where`, `$out`, `$merge`, or anything server-JS.
11. Output MUST be parseable by `json.loads` — double-quoted keys/strings only.
12. The `enrichment` block on each chart is the SAFETY NET — even if the
    pipeline already projects names, declare the (column → collection → label)
    mapping for any column that *could* contain raw IDs at runtime, so the
    backend can resolve them post-execution.

Return the JSON now.
"""


class MongoVanna:
    """
    Vanna-AI-style engine for MongoDB.

    Train once at startup (introspects schema and seeds examples).  Then call
    ``ask(question)`` to get a multi-chart dashboard plan + an HTML template.
    The pipelines are NOT executed during ``ask`` — execution is the caller's
    responsibility (see ``run()`` for a helper).
    """

    def __init__(
        self,
        db: Database,
        llm_client: LLMClient | None = None,
        max_examples: int = 16,
    ) -> None:
        self._db = db
        self._llm = llm_client or get_dashboard_tool_client()
        self._max_examples = max_examples

        # Training data
        self._schema: dict[str, Any] = {}
        self._docs: list[str] = []
        self._examples: list[dict[str, Any]] = []
        self._trained = False

    # ── Training API ──────────────────────────────────────────────────────────

    def train(
        self,
        ddl: str | None = None,
        documentation: str | None = None,
        question: str | None = None,
        collection: str | None = None,
        pipeline: list[dict] | None = None,
    ) -> None:
        if ddl:
            self._docs.append(f"[schema] {ddl}")
        if documentation:
            self._docs.append(f"[doc] {documentation}")
        if question and collection and pipeline is not None:
            self._examples.append(
                {"question": question, "collection": collection, "pipeline": pipeline}
            )

    def train_from_schema(self, sample_size: int = 8) -> dict[str, Any]:
        """Introspect the live database and load it as training context."""
        self._schema = introspect_schema(self._db, sample_size=sample_size)

        known = set(self._schema.get("fields", {}).keys())
        injected = 0
        for ex in SEED_EXAMPLES:
            if ex["collection"] in known and ex not in self._examples:
                self._examples.append(ex)
                injected += 1

        if SCHEMA_DOCUMENTATION and not any(
            d.startswith("[doc] schema-glossary") for d in self._docs
        ):
            self._docs.append(f"[doc] schema-glossary\n{SCHEMA_DOCUMENTATION}")

        self._trained = True
        logger.info(
            "MongoVanna trained from live schema",
            collections=list(known),
            collection_count=len(known),
            references_detected=sum(
                len(v) for v in self._schema.get("references", {}).values()
            ),
            seed_examples_injected=injected,
        )
        return self._schema

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── Public ask() ─────────────────────────────────────────────────────────

    async def ask(self, question: str) -> dict[str, Any]:
        """
        Translate `question` into one or more MongoDB pipelines and a polished
        dashboard HTML template.  Always returns a dict (never raises).
        """
        if not self._trained:
            self.train_from_schema()

        try:
            plan = await self._generate_plan(question)
        except Exception as exc:
            logger.error(
                "MongoVanna: plan generation failed",
                error=str(exc),
                exc_info=True,
            )
            return {
                "success": False,
                "stage": "generate_plan",
                "error": str(exc),
                "question": question,
            }

        dashboard_id = f"viz-{uuid.uuid4().hex[:8]}"
        for idx, chart in enumerate(plan["charts"]):
            chart["id"] = f"{dashboard_id}-c{idx + 1}"
            chart["pipeline"] = _json_safe(chart.get("pipeline", []))

        dashboard_html = self._build_dashboard_html(dashboard_id, plan)

        first = plan["charts"][0]

        return {
            "success":         True,
            "question":        question,
            "dashboard_id":    dashboard_id,
            "dashboard_title": plan.get("dashboard_title", ""),
            "dashboard_description": plan.get("dashboard_description", ""),
            "theme":           plan.get("theme", "dark"),
            "explanation":     plan.get("explanation", ""),
            "charts":          plan["charts"],
            "dashboard_html":  dashboard_html,
            "chart_data_placeholder": CHART_DATA_PLACEHOLDER,

            # Legacy single-chart fields
            "collection":     first.get("collection"),
            "pipeline":       first.get("pipeline"),
            "output_columns": first.get("output_columns", []),
            "chart_type":     first.get("chart_type", "table"),
            "chart_x":        first.get("chart_x"),
            "chart_y":        first.get("chart_y"),
            "chart_title":    first.get("chart_title", ""),
            "chart_description": first.get("chart_description", ""),
            "chart_html":     dashboard_html,
            "training_examples_used": len(self._examples),
        }

    # ── Pipeline execution ────────────────────────────────────────────────────

    def run(
        self,
        collection: str,
        pipeline: list[dict],
        hard_cap: int | None = None,
        enrichment: list[dict[str, str]] | None = None,
    ) -> list[dict]:
        fields_schema = self._schema.get("fields", {}) if isinstance(self._schema, dict) else {}
        if collection not in fields_schema:
            self.train_from_schema()
            fields_schema = self._schema.get("fields", {})
            if collection not in fields_schema:
                raise ValueError(f"Unknown collection '{collection}'")

        capped = list(pipeline)
        if hard_cap is not None and not any("$limit" in stage for stage in capped):
            capped.append({"$limit": hard_cap})

        cursor = self._db[collection].aggregate(capped, allowDiskUse=True)
        rows = [_json_safe(doc) for doc in cursor]

        if enrichment:
            self._enrich_rows(rows, enrichment)

        return rows

    # ── ID enrichment ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_dotted(doc: Any, dotted: str) -> Any:
        cur: Any = doc
        for part in dotted.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
            if cur is None:
                return None
        return cur

    @classmethod
    def _extract_label(cls, doc: dict, primary: str) -> str | None:
        candidates: list[Any] = []
        first = doc.get("firstName") or doc.get("first_name")
        last = doc.get("lastName") or doc.get("last_name")
        if first and last:
            composed = f"{first} {last}".strip()
            if composed:
                candidates.append(composed)

        if primary:
            v = cls._resolve_dotted(doc, primary)
            candidates.append(v)
            if v is None and "." in primary:
                base = primary.rsplit(".", 1)[0]
                base_val = cls._resolve_dotted(doc, base)
                if isinstance(base_val, dict):
                    for loc in ("en", "ar", "fr", "es", "default", "value"):
                        if base_val.get(loc):
                            candidates.append(base_val[loc])
                            break

        for key in ("name", "title", "fullName", "displayName", "username", "code"):
            v = doc.get(key)
            if isinstance(v, dict):
                for loc in ("en", "ar", "fr", "es", "default", "value"):
                    if v.get(loc):
                        candidates.append(v[loc])
                        break
            else:
                candidates.append(v)

        if first and not last:
            candidates.append(str(first))
        elif last and not first:
            candidates.append(str(last))

        for c in candidates:
            if c is None:
                continue
            if isinstance(c, str) and c.strip():
                return c
            if isinstance(c, (int, float)):
                return str(c)
        return None

    @staticmethod
    def _looks_like_objectid(value: Any) -> bool:
        if isinstance(value, ObjectId):
            return True
        if isinstance(value, str) and len(value) == 24:
            try:
                int(value, 16)
                return True
            except ValueError:
                return False
        return False

    def _enrich_rows(
        self,
        rows: list[dict],
        enrichment: list[dict[str, str]],
    ) -> None:
        if not rows or not enrichment:
            return

        for rule in enrichment:
            column = rule.get("column")
            ref_col = rule.get("lookup_collection")
            label_field = rule.get("label_field") or "name"
            if not column or not ref_col:
                continue
            if ref_col not in self._db.list_collection_names():
                continue

            candidate_ids: set[Any] = set()
            for row in rows:
                v = row.get(column)
                if self._looks_like_objectid(v):
                    candidate_ids.add(v)
            if not candidate_ids:
                continue

            id_filter: list[Any] = []
            for cid in candidate_ids:
                try:
                    id_filter.append(ObjectId(cid))
                except Exception:
                    id_filter.append(cid)

            try:
                cursor = self._db[ref_col].find({"_id": {"$in": id_filter}})
                lookup: dict[str, str] = {}
                for doc in cursor:
                    label = self._extract_label(doc, label_field)
                    if label is not None:
                        lookup[str(doc.get("_id"))] = label
            except Exception as exc:
                logger.warning(
                    "Enrichment lookup failed",
                    column=column,
                    ref_col=ref_col,
                    error=str(exc),
                )
                continue

            for row in rows:
                v = row.get(column)
                if self._looks_like_objectid(v):
                    label = lookup.get(str(v))
                    if label:
                        row[column] = label

    # ── Plan generation ───────────────────────────────────────────────────────

    async def _generate_plan(self, question: str) -> dict[str, Any]:
        schema_block = schema_to_prompt_text(self._schema)
        examples_block = self._render_examples()
        docs_block = self._render_docs()

        user_msg = (
            f"DATABASE SCHEMA\n----------------\n{schema_block}\n\n"
            f"{docs_block}"
            f"{examples_block}"
            f"USER QUESTION\n--------------\n{question}\n\n"
            "Return the JSON plan now."
        )

        response = await self._llm.ainvoke_with_retry(
            [SystemMessage(content=_PLAN_SYSTEM_PROMPT), HumanMessage(content=user_msg)]
        )
        raw = _strip_fences(response.content)

        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as exc:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                plan = json.loads(raw[start : end + 1])
            else:
                raise ValueError(f"LLM did not return valid JSON: {exc}") from exc

        if "charts" not in plan and "collection" in plan:
            plan = {
                "dashboard_title": plan.get("dashboard_title")
                or plan.get("chart_title")
                or plan.get("title")
                or "Analytics",
                "dashboard_description": plan.get("dashboard_description")
                or plan.get("description")
                or plan.get("explanation", ""),
                "explanation": plan.get("explanation", ""),
                "theme": plan.get("theme", "dark"),
                "charts": [plan],
            }

        if not isinstance(plan.get("charts"), list) or not plan["charts"]:
            raise ValueError("LLM plan must contain a non-empty 'charts' array")

        known_collections = set(self._schema.get("fields", {}).keys())
        for chart in plan["charts"]:
            self._validate_and_normalise_chart(chart, known_collections, question)

        self._normalise_dashboard_metadata(plan, question)
        plan.setdefault("explanation", plan.get("dashboard_description", ""))
        theme = plan.get("theme", "dark")
        plan["theme"] = theme if theme in {"light", "dark"} else "dark"

        return plan

    def _validate_and_normalise_chart(
        self,
        chart: dict[str, Any],
        known_collections: set[str],
        question: str,
    ) -> None:
        for required in ("collection", "pipeline"):
            if required not in chart:
                raise ValueError(
                    f"Chart spec missing required key '{required}': {chart}"
                )

        if chart["collection"] not in known_collections:
            raise ValueError(
                f"LLM picked unknown collection '{chart['collection']}'. "
                f"Known: {sorted(known_collections)}"
            )

        if not isinstance(chart["pipeline"], list):
            raise ValueError("Pipeline is not a list")

        ct = chart.get("chart_type", "table")
        if ct not in _VALID_CHART_TYPES:
            ct = "bar"
        chart["chart_type"] = ct

        cols = chart.get("output_columns") or []
        chart["output_columns"] = list(cols)
        if cols:
            if chart.get("chart_x") not in cols:
                chart["chart_x"] = cols[0]
            if chart.get("chart_y") not in cols and len(cols) > 1:
                chart["chart_y"] = cols[1]

        chart_title = _clean_metadata_text(
            chart.get("chart_title") or chart.get("title")
        )
        chart["chart_title"] = chart_title or self._generated_chart_title(chart)

        chart_description = _clean_metadata_text(
            chart.get("chart_description")
            or chart.get("description")
            or chart.get("chart_subtitle")
            or chart.get("subtitle")
        )
        chart["chart_description"] = (
            chart_description
            or self._generated_chart_description(chart, question)
        )
        chart["chart_subtitle"] = _clean_metadata_text(
            chart.get("chart_subtitle") or chart.get("subtitle")
        )
        chart.setdefault("x_axis_label", chart.get("chart_x", ""))
        chart.setdefault("y_axis_label", chart.get("chart_y", ""))
        chart["colors"] = _resolve_palette(chart.get("colors"))

        size = chart.get("size", "md")
        chart["size"] = size if size in {"sm", "md", "lg", "full"} else "md"

        sort_dir = chart.get("sort_dir", "desc")
        chart["sort_dir"] = sort_dir if sort_dir in {"asc", "desc"} else "desc"

        if not isinstance(chart.get("enrichment"), list):
            chart["enrichment"] = []
        chart["stacked"] = bool(chart.get("stacked", False))

    def _generated_chart_title(self, chart: dict[str, Any]) -> str:
        collection = _humanise_identifier(chart.get("collection"))
        x_label = _humanise_identifier(chart.get("chart_x"))
        y_label = _humanise_identifier(chart.get("chart_y"))
        chart_type = chart.get("chart_type", "table")

        if chart_type == "table":
            return f"{collection or 'Data'} Detail Table"
        if x_label and y_label:
            return f"{y_label} by {x_label}"
        if y_label:
            return f"{y_label} Overview"
        if x_label:
            return f"{x_label} Breakdown"
        return (
            f"{collection or 'Analytics'} "
            f"{_humanise_identifier(chart_type) or 'Chart'}"
        )

    def _generated_chart_description(
        self,
        chart: dict[str, Any],
        question: str,
    ) -> str:
        collection = _humanise_identifier(chart.get("collection")).lower()
        x_label = _humanise_identifier(chart.get("chart_x")).lower()
        y_label = _humanise_identifier(chart.get("chart_y")).lower()
        chart_label = _chart_type_label(chart.get("chart_type", "table"))
        intent = _summarise_question_intent(question)

        if x_label and y_label:
            sentence = f"Shows {y_label} by {x_label}"
        elif y_label:
            sentence = f"Shows {y_label} values"
        elif x_label:
            sentence = f"Shows a breakdown by {x_label}"
        else:
            sentence = f"Shows the requested {chart_label}"

        if collection:
            sentence += f" from the {collection} collection"
        if intent:
            sentence += f" for {intent}"
        return sentence.rstrip(" .") + "."

    def _normalise_dashboard_metadata(
        self,
        plan: dict[str, Any],
        question: str,
    ) -> None:
        title = _clean_metadata_text(
            plan.get("dashboard_title") or plan.get("title")
        )
        plan["dashboard_title"] = title or self._generated_dashboard_title(
            plan, question
        )

        description = _clean_metadata_text(
            plan.get("dashboard_description")
            or plan.get("description")
            or plan.get("explanation")
        )
        plan["dashboard_description"] = (
            description or self._generated_dashboard_description(plan, question)
        )
        plan["explanation"] = (
            _clean_metadata_text(plan.get("explanation"))
            or plan["dashboard_description"]
        )

    def _generated_dashboard_title(
        self,
        plan: dict[str, Any],
        question: str,
    ) -> str:
        intent = _summarise_question_intent(question)
        if intent:
            title = _title_case_phrase(intent, max_words=9)
            if not re.search(r"\bdashboard\b", title, flags=re.IGNORECASE):
                title = f"{title} Dashboard"
            return title

        chart_titles = [
            _clean_metadata_text(chart.get("chart_title"))
            for chart in plan.get("charts", [])
        ]
        chart_titles = [title for title in chart_titles if title]
        if len(chart_titles) == 1:
            return f"{chart_titles[0]} Dashboard"
        return "Analytics Dashboard"

    def _generated_dashboard_description(
        self,
        plan: dict[str, Any],
        question: str,
    ) -> str:
        intent = _summarise_question_intent(question)
        chart_titles = [
            _clean_metadata_text(chart.get("chart_title"))
            for chart in plan.get("charts", [])
        ]
        chart_titles = [title for title in chart_titles if title]
        if len(chart_titles) > 3:
            chart_titles = chart_titles[:3] + [f"{len(chart_titles) - 3} more views"]
        chart_summary = _join_human_list(chart_titles)

        if chart_summary and intent:
            return f"Combines {chart_summary} to answer {intent}."
        if chart_summary:
            return f"Combines {chart_summary} into a self-contained analytics dashboard."
        if intent:
            return f"Summarizes the requested analysis for {intent}."
        return "Summarizes the requested data in a self-contained analytics dashboard."

    def _render_examples(self) -> str:
        if not self._examples:
            return ""
        sample = self._examples[-self._max_examples:]
        parts = ["TRAINING EXAMPLES\n-----------------"]
        for ex in sample:
            parts.append(
                f"Q: {ex['question']}\n"
                f"  collection: {ex['collection']}\n"
                f"  pipeline:   {json.dumps(ex['pipeline'])}"
            )
        parts.append("")
        return "\n".join(parts) + "\n"

    def _render_docs(self) -> str:
        if not self._docs:
            return ""
        return (
            "BUSINESS DOCUMENTATION\n----------------------\n"
            + "\n\n".join(self._docs)
            + "\n\n"
        )

    # ── HTML rendering ────────────────────────────────────────────────────────

    def _build_dashboard_html(
        self,
        dashboard_id: str,
        plan: dict[str, Any],
    ) -> str:
        """
        Polished, self-contained dashboard HTML.

        Includes dashboard-level context, chart-level context, and a minimal footer.
        Cards are staggered by index for a smooth entrance animation.
        """
        theme = plan.get("theme", "dark")
        if theme not in {"light", "dark"}:
            theme = "dark"

        charts_payload = [self._chart_runtime_spec(chart) for chart in plan["charts"]]
        chart_cards    = [self._render_chart_card(chart)   for chart in plan["charts"]]

        # Inject per-card stagger delay
        staggered = []
        for i, card_html in enumerate(chart_cards):
            delay = f"{i * 0.07:.2f}s"
            card_html = card_html.replace(
                'data-mv-chart=',
                f'style="animation-delay:{delay}" data-mv-chart=',
                1,
            )
            staggered.append(card_html)

        cards_html  = "\n".join(staggered)
        charts_json = json.dumps(charts_payload)

        css = _DASHBOARD_CSS
        js  = _DASHBOARD_JS.replace("__DASHBOARD_ID__", dashboard_id)

        theme_cls = "mv-theme-light" if theme == "light" else ""
        next_theme = "dark" if theme == "light" else "light"
        dashboard_title = html_lib.escape(
            plan.get("dashboard_title") or "Analytics Dashboard"
        )
        dashboard_description = html_lib.escape(
            plan.get("dashboard_description") or plan.get("explanation") or ""
        )
        dashboard_desc_html = (
            f'<p class="mv-dashboard-description">{dashboard_description}</p>'
            if dashboard_description
            else ""
        )

        html = (
            f'<div id="{dashboard_id}" class="mv-root {theme_cls}" '
            f'data-mv-dashboard="{dashboard_id}" data-mv-theme="{theme}">\n'
            f"<style>{css}</style>\n"
            f'<header class="mv-dashboard-header">'
            f'<div class="mv-dashboard-copy">'
            f'<p class="mv-dashboard-label">Dashboard</p>'
            f'<h2 class="mv-dashboard-title">{dashboard_title}</h2>'
            f"{dashboard_desc_html}"
            f"</div>"
            f'<div class="mv-toolbar">'
            f'<button type="button" class="mv-theme-toggle" '
            f'data-mv-action="theme" aria-label="Switch to {next_theme} theme">'
            f'<span class="mv-theme-toggle-icon" aria-hidden="true"></span>'
            f'<span data-mv-theme-label>{next_theme.title()}</span>'
            f"</button>"
            f"</div>\n"
            f"</header>\n"
            f'<div class="mv-grid">\n{cards_html}\n</div>\n'
            f'<footer class="mv-footer">'
            f"</footer>\n"
            f"</div>\n"
            f"<script>\n"
            f"window.MongoVannaCharts = window.MongoVannaCharts || {{}};\n"
            f'window.MongoVannaCharts["{dashboard_id}"] = {charts_json};\n'
            f"{js}\n"
            f"</script>"
        )
        return html.replace("__CHART_DATA_PLACEHOLDER__", CHART_DATA_PLACEHOLDER)

    def _render_chart_card(self, chart: dict[str, Any]) -> str:
        """
        Render one card section.

        Title + description are baked into the HTML so the dashboard is
        readable before JS injects live data.
        """
        cid      = chart["id"]
        ct       = chart["chart_type"]
        size_cls = f"mv-card-{chart.get('size', 'md')}"
        title    = html_lib.escape(chart.get("chart_title", ""))
        subtitle = html_lib.escape(chart.get("chart_subtitle", ""))
        description = html_lib.escape(chart.get("chart_description", ""))
        badge    = ct.upper()

        if title:
            sub_html = (
                f'<p class="mv-card-subtitle">{subtitle}</p>'
                if subtitle and subtitle != description
                else ""
            )
            desc_html = (
                f'<p class="mv-card-description">{description}</p>'
                if description
                else ""
            )
            header = (
                f'<div class="mv-card-header">'
                f'<div><p class="mv-card-title">{title}</p>{sub_html}{desc_html}</div>'
                f'<span class="mv-chip">{badge}</span>'
                f"</div>"
            )
        else:
            header = ""

        if ct == "table":
            body = (
                '<div class="mv-table-wrap">'
                '<table class="mv-table"><thead></thead><tbody></tbody></table>'
                "</div>"
            )
        else:
            body = (
                '<div class="mv-canvas-wrap">'
                '<canvas></canvas>'
                "</div>"
            )

        return (
            f'<section class="mv-card {size_cls}" data-mv-chart="{cid}">\n'
            f"  {header}\n"
            f'  <div class="mv-card-body">{body}</div>\n'
            f'  <div class="mv-card-empty">No data yet</div>\n'
            f"</section>"
        )

    def _chart_runtime_spec(self, chart: dict[str, Any]) -> dict[str, Any]:
        """The minimal JS-side spec the dashboard JS needs to render a chart."""
        return {
            "id":            chart["id"],
            "chartType":     chart["chart_type"],
            "x":             chart.get("chart_x") or "",
            "y":             chart.get("chart_y") or "",
            "title":         chart.get("chart_title", ""),
            "description":   chart.get("chart_description", ""),
            "subtitle":      chart.get("chart_subtitle", ""),
            "xLabel":        chart.get("x_axis_label") or chart.get("chart_x") or "",
            "yLabel":        chart.get("y_axis_label") or chart.get("chart_y") or "",
            "colors":        chart.get("colors") or PALETTES["default"],
            "stacked":       bool(chart.get("stacked", False)),
            "outputColumns": chart.get("output_columns", []),
            "sortBy":        chart.get("sort_by"),
            "sortDir":       chart.get("sort_dir", "desc"),
        }


# ── Static CSS / JS bundle ───────────────────────────────────────────────────

_DASHBOARD_CSS = r"""
/* ── MongoVanna v2 — Obsidian Dark ───────────────────────────────────────── */
.mv-root{
  --mv-bg:#070d1a;--mv-surface:rgba(255,255,255,.03);
  --mv-surface-h:rgba(255,255,255,.055);--mv-border:rgba(255,255,255,.07);
  --mv-border-h:rgba(255,255,255,.16);--mv-text:#e2e8f0;--mv-muted:#64748b;
  --mv-mid:#94a3b8;--mv-accent:#818cf8;--mv-accent2:#06b6d4;
  --mv-accent3:#34d399;--mv-radius:14px;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
    Helvetica,Arial,sans-serif;
  background:var(--mv-bg);color:var(--mv-text);
  padding:20px;border-radius:16px;width:100%;box-sizing:border-box;
}
.mv-root.mv-theme-light{
  --mv-bg:#f1f5f9;--mv-surface:#ffffff;--mv-surface-h:#f8fafc;
  --mv-border:rgba(15,23,42,.08);--mv-border-h:rgba(15,23,42,.2);
  --mv-text:#0f172a;--mv-muted:#64748b;--mv-mid:#475569;
  --mv-accent:#6366f1;--mv-accent2:#0ea5e9;--mv-accent3:#10b981;
}
.mv-root *{box-sizing:border-box;}
.mv-toolbar{
  display:flex;justify-content:flex-end;align-items:center;
  margin:0;min-height:32px;flex-shrink:0;
}
.mv-theme-toggle{
  display:inline-flex;align-items:center;gap:8px;height:32px;
  padding:0 10px;border:1px solid var(--mv-border);
  border-radius:999px;background:var(--mv-surface);color:var(--mv-text);
  font:inherit;font-size:11px;font-weight:700;letter-spacing:0;
  text-transform:uppercase;cursor:pointer;
  transition:border-color .2s ease,background .2s ease,transform .18s ease;
}
.mv-theme-toggle:hover{
  background:var(--mv-surface-h);border-color:var(--mv-border-h);
  transform:translateY(-1px);
}
.mv-theme-toggle:focus-visible{
  outline:2px solid var(--mv-accent);outline-offset:2px;
}
.mv-theme-toggle-icon{
  width:14px;height:14px;border-radius:50%;flex:0 0 auto;
  background:linear-gradient(135deg,#f59e0b,#facc15);
  box-shadow:0 0 0 3px rgba(245,158,11,.16);
}
.mv-root.mv-theme-light .mv-theme-toggle-icon{
  background:linear-gradient(135deg,#818cf8,#06b6d4);
  box-shadow:0 0 0 3px rgba(129,140,248,.14);
}
.mv-dashboard-header{
  display:flex;justify-content:space-between;align-items:flex-start;
  gap:20px;margin-bottom:18px;
}
.mv-dashboard-copy{min-width:0;max-width:780px;}
.mv-dashboard-label{
  margin:0 0 7px;font-size:10px;font-weight:700;letter-spacing:0;
  color:var(--mv-accent2);text-transform:uppercase;
}
.mv-dashboard-title{
  margin:0;color:var(--mv-text);font-size:24px;font-weight:700;
  line-height:1.16;letter-spacing:0;overflow-wrap:anywhere;
}
.mv-dashboard-description{
  margin:8px 0 0;color:var(--mv-mid);font-size:13px;
  line-height:1.55;overflow-wrap:anywhere;
}
.mv-grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;}
.mv-card{
  grid-column:span 6;background:var(--mv-surface);
  border:1px solid var(--mv-border);border-radius:var(--mv-radius);
  padding:20px;display:flex;flex-direction:column;min-height:300px;
  position:relative;overflow:hidden;
  transition:border-color .2s ease,background .2s ease,transform .22s ease;
  animation:mv-card-in .5s ease both;
}
@keyframes mv-card-in{
  from{opacity:0;transform:translateY(14px);}
  to{opacity:1;transform:none;}
}
.mv-card:hover{
  background:var(--mv-surface-h);border-color:var(--mv-border-h);
  transform:translateY(-2px);
}
.mv-card-sm  {grid-column:span 4;min-height:260px;}
.mv-card-md  {grid-column:span 6;}
.mv-card-lg  {grid-column:span 8;min-height:380px;}
.mv-card-full{grid-column:span 12;min-height:380px;}
@media(max-width:880px){
  .mv-dashboard-header{flex-direction:column;gap:12px;}
  .mv-toolbar{align-self:flex-start;}
  .mv-dashboard-title{font-size:20px;}
  .mv-card,.mv-card-sm,.mv-card-md,
  .mv-card-lg,.mv-card-full{grid-column:span 12;}
}
.mv-card-header{
  display:flex;justify-content:space-between;align-items:flex-start;
  gap:12px;margin-bottom:16px;flex-shrink:0;
}
.mv-card-header>div{min-width:0;}
.mv-card-title{
  margin:0;font-size:13px;font-weight:600;color:var(--mv-text);
  letter-spacing:0;line-height:1.3;overflow-wrap:anywhere;
}
.mv-card-subtitle{
  margin:3px 0 0;font-size:11px;color:var(--mv-muted);line-height:1.4;
  overflow-wrap:anywhere;
}
.mv-card-description{
  margin:6px 0 0;font-size:12px;color:var(--mv-mid);line-height:1.5;
  overflow-wrap:anywhere;
}
.mv-chip{
  font-size:10px;font-weight:700;letter-spacing:0;text-transform:uppercase;
  color:var(--mv-accent);background:rgba(129,140,248,.1);
  padding:3px 9px;border-radius:999px;flex-shrink:0;white-space:nowrap;
}
.mv-card-body{flex:1;position:relative;min-height:0;}
.mv-canvas-wrap{position:relative;width:100%;height:100%;min-height:220px;}
.mv-canvas-wrap canvas{position:absolute;inset:0;width:100%!important;height:100%!important;}
.mv-table-wrap{
  overflow:auto;border-radius:10px;
  border:1px solid var(--mv-border);height:100%;max-height:320px;
}
.mv-table{width:100%;border-collapse:collapse;font-size:12px;}
.mv-table th{
  padding:9px 13px;background:rgba(148,163,184,.05);color:var(--mv-muted);
  font-weight:600;text-align:left;position:sticky;top:0;
  border-bottom:1px solid var(--mv-border);
  font-size:10px;letter-spacing:0;text-transform:uppercase;z-index:1;
}
.mv-table td{
  padding:8px 13px;
  border-bottom:1px solid rgba(148,163,184,.05);color:var(--mv-text);
}
.mv-table tr:last-child td{border-bottom:none;}
.mv-table tbody tr:hover td{background:rgba(129,140,248,.05);}
.mv-card-empty{
  display:none;text-align:center;color:var(--mv-muted);
  font-size:13px;padding:32px 16px;
}
.mv-card.mv-empty .mv-card-body{display:none;}
.mv-card.mv-empty .mv-card-empty{display:block;}
.mv-footer{margin-top:20px;text-align:center;}
.mv-footer-text{
  font-size:10px;color:rgba(148,163,184,.3);
  letter-spacing:0;text-transform:uppercase;
}
.mv-root.mv-theme-light .mv-footer-text{color:rgba(71,85,105,.42);}
"""

_DASHBOARD_JS = r"""
(function(){
  var root = document.getElementById("__DASHBOARD_ID__");
  if (!root || root.__mvInited) return;
  root.__mvInited = true;

  var specs = (window.MongoVannaCharts || {})["__DASHBOARD_ID__"] || [];
  var chartInstances = {};

  function ensureChartJs(cb){
    if (window.Chart) return cb();
    var s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/chart.js@4";
    s.onload = cb;
    s.onerror = function(){ console.error("MongoVanna: Chart.js failed to load"); };
    document.head.appendChild(s);
  }

  var GRID_COLOR, TICK_COLOR, BG_COLOR, TOOLTIP_STYLE;
  var lastRows = {};

  function syncThemeVars(){
    var isLight = root.classList.contains("mv-theme-light");
    GRID_COLOR = isLight ? "rgba(15,23,42,.07)" : "rgba(148,163,184,.07)";
    TICK_COLOR = isLight ? "#475569" : "#94a3b8";
    BG_COLOR   = isLight ? "#f1f5f9" : "#070d1a";
    TOOLTIP_STYLE = isLight
      ? {backgroundColor:"#ffffff",titleColor:"#0f172a",bodyColor:"#475569",
         borderColor:"rgba(15,23,42,.12)"}
      : {backgroundColor:"#1e293b",titleColor:"#f1f5f9",bodyColor:"#94a3b8",
         borderColor:"rgba(255,255,255,.09)"};
  }

  syncThemeVars();

  function hexToRgba(hex, a){
    var h = (hex || "#818cf8").replace("#","");
    if (h.length === 3) h = h.split("").map(function(c){return c+c;}).join("");
    return "rgba("+parseInt(h.slice(0,2),16)+","+parseInt(h.slice(2,4),16)+","+parseInt(h.slice(4,6),16)+","+a+")";
  }

  function sortRows(rows, sortBy, dir, xCol, yCol){
    if (!sortBy) return rows;
    var col = sortBy === "x" ? xCol : (sortBy === "y" ? yCol : sortBy);
    if (!col) return rows;
    var sign = dir === "asc" ? 1 : -1;
    return rows.slice().sort(function(a,b){
      var av = a[col], bv = b[col];
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * sign;
      return String(av).localeCompare(String(bv)) * sign;
    });
  }

  function renderTable(card, spec, rows){
    var thead = card.querySelector("thead");
    var tbody = card.querySelector("tbody");
    if (!thead || !tbody) return;
    var cols = (spec.outputColumns && spec.outputColumns.length)
                 ? spec.outputColumns
                 : (rows[0] ? Object.keys(rows[0]) : []);
    thead.innerHTML = "<tr>" + cols.map(function(c){
      return "<th>" + c + "</th>";
    }).join("") + "</tr>";
    tbody.innerHTML = rows.map(function(r){
      return "<tr>" + cols.map(function(c){
        var v = r[c];
        return "<td>" + (v === null || v === undefined ? "" : String(v)) + "</td>";
      }).join("") + "</tr>";
    }).join("");
  }

  function renderChart(card, spec, rows){
    var canvas = card.querySelector("canvas");
    if (!canvas) return;
    if (chartInstances[spec.id]){
      try { chartInstances[spec.id].destroy(); } catch(e){}
      chartInstances[spec.id] = null;
    }

    var sorted  = sortRows(rows, spec.sortBy, spec.sortDir, spec.x, spec.y);
    var labels  = sorted.map(function(r){ return r[spec.x]; });
    var values  = sorted.map(function(r){
      var v = r[spec.y];
      return (v === null || v === undefined) ? null : Number(v);
    });
    var palette = (spec.colors && spec.colors.length)
                    ? spec.colors
                    : ["#818cf8","#06b6d4","#34d399","#fb923c","#f472b6","#facc15"];

    var cjsType = spec.chartType;
    var fill    = false;
    if (cjsType === "area"){ cjsType = "line"; fill = true; }

    var multiColor = (cjsType==="pie"||cjsType==="doughnut"||cjsType==="polarArea");

    var bgColor, borderColor;
    if (multiColor){
      bgColor     = labels.map(function(_,i){ return palette[i % palette.length]; });
      borderColor = bgColor;
    } else if (cjsType === "bar"){
      bgColor     = labels.map(function(_,i){ return hexToRgba(palette[i % palette.length], .78); });
      borderColor = labels.map(function(_,i){ return palette[i % palette.length]; });
    } else {
      bgColor = fill ? function(ctx){
        var g = ctx.chart.ctx.createLinearGradient(0,0,0,200);
        g.addColorStop(0, hexToRgba(palette[0], .28));
        g.addColorStop(1, hexToRgba(palette[0], 0));
        return g;
      } : hexToRgba(palette[0], .8);
      borderColor = palette[0];
    }

    syncThemeVars();
    var TT = Object.assign({
      cornerRadius:10,padding:12,boxPadding:5,borderWidth:1,
    }, TOOLTIP_STYLE);
    var AX = {
      grid:{color:GRID_COLOR},ticks:{color:TICK_COLOR,font:{size:10}},
      border:{color:"transparent"},
    };

    var dataset = {
      label:               spec.yLabel || spec.y || "",
      data:                values,
      backgroundColor:     bgColor,
      borderColor:         borderColor,
      borderWidth:         2,
      fill:                fill,
      tension:             0.4,
      pointRadius:         cjsType==="scatter" ? 5 : (cjsType==="line" ? 3 : 0),
      pointHoverRadius:    cjsType==="scatter" ? 7 : 5,
      pointBackgroundColor:palette[0],
      pointBorderColor:    BG_COLOR,
      pointBorderWidth:    2,
      borderRadius:        cjsType==="bar" ? 8 : 0,
      borderSkipped:       false,
      hoverOffset:         (cjsType==="doughnut"||cjsType==="pie") ? 6 : 0,
    };

    var data = (cjsType==="scatter")
      ? { datasets:[{ label:dataset.label,
            data:sorted.map(function(r){return{x:Number(r[spec.x]),y:Number(r[spec.y]);};}),
            backgroundColor:bgColor, borderColor:borderColor, pointRadius:5 }] }
      : { labels:labels, datasets:[dataset] };

    var showLegend = multiColor || cjsType==="radar";
    var options = {
      responsive:true, maintainAspectRatio:false,
      animation:{duration:800, easing:"easeOutQuart"},
      plugins:{
        legend:{
          display:showLegend, position:"bottom",
          labels:{color:TICK_COLOR,boxWidth:10,padding:14,font:{size:10}},
        },
        tooltip:Object.assign({},TT,
          (spec.chartType==="area"||cjsType==="line")
            ? {mode:"index",intersect:false} : {}),
      },
      scales:(multiColor||cjsType==="radar") ? {} : {
        x:Object.assign({},AX,{
          stacked:!!spec.stacked,
          title:{display:!!spec.xLabel,text:spec.xLabel,color:TICK_COLOR},
          ticks:Object.assign({},AX.ticks,{autoSkip:true,maxTicksLimit:12,maxRotation:45,minRotation:0}),
        }),
        y:Object.assign({},AX,{
          stacked:!!spec.stacked, beginAtZero:true,
          title:{display:!!spec.yLabel,text:spec.yLabel,color:TICK_COLOR},
        }),
      },
    };
    if (cjsType==="doughnut"||cjsType==="pie") options.cutout = "70%";

    chartInstances[spec.id] = new Chart(canvas.getContext("2d"),{type:cjsType,data:data,options:options});
  }

  function renderOne(spec, rows){
    var card = root.querySelector('[data-mv-chart="'+spec.id+'"]');
    if (!card) return;
    rows = rows || [];
    var hasData = rows.length > 0;
    card.classList.toggle("mv-empty", !hasData);

    if      (spec.chartType==="table") renderTable(card, spec, rows);
    else if (hasData)                  ensureChartJs(function(){ renderChart(card, spec, rows); });
  }

  function refresh(allRows){
    allRows = allRows || {};
    syncThemeVars();
    specs.forEach(function(spec){ renderOne(spec, allRows[spec.id] || []); });
  }

  function updateThemeToggle(){
    var btn = root.querySelector('[data-mv-action="theme"]');
    if (!btn) return;
    var isLight = root.classList.contains("mv-theme-light");
    var nextTheme = isLight ? "dark" : "light";
    var label = btn.querySelector("[data-mv-theme-label]");
    if (label) label.textContent = nextTheme.charAt(0).toUpperCase() + nextTheme.slice(1);
    btn.setAttribute("aria-label", "Switch to " + nextTheme + " theme");
    btn.setAttribute("aria-pressed", isLight ? "true" : "false");
  }

  function setTheme(theme){
    var isLight = theme === "light";
    root.classList.toggle("mv-theme-light", isLight);
    root.dataset.mvTheme = isLight ? "light" : "dark";
    // Force a reflow to trigger CSS variable re-computation
    void root.offsetHeight;
    syncThemeVars();
    updateThemeToggle();
    refresh(lastRows);
  }

  var themeBtn = root.querySelector('[data-mv-action="theme"]');
  if (themeBtn){
    updateThemeToggle();
    themeBtn.addEventListener("click", function(){
      setTheme(root.classList.contains("mv-theme-light") ? "dark" : "light");
    });
  }

  var initial = __CHART_DATA_PLACEHOLDER__;
  lastRows = initial || {};
  refresh(lastRows);

  window.MongoVannaDashboards = window.MongoVannaDashboards || {};
  window.MongoVannaDashboards["__DASHBOARD_ID__"] = {
    refresh:function(rows){
      lastRows = rows || {};
      window.MongoVannaDashboards["__DASHBOARD_ID__"]._lastRows = lastRows;
      refresh(lastRows);
    },
    setTheme:setTheme,
    toggleTheme:function(){
      setTheme(root.classList.contains("mv-theme-light") ? "dark" : "light");
    },
    getTheme:function(){
      return root.classList.contains("mv-theme-light") ? "light" : "dark";
    },
    specs    : specs,
    _lastRows: lastRows,
  };
})();
"""
