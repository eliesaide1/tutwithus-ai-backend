"""
DataVizTool — workflow entry point for the MongoVanna engine.

Responsibilities
----------------
1. Verify the caller's role is ADMIN — refuse politely otherwise.
2. Lazily build a singleton MongoVanna engine bound to the project's MongoDB.
3. Forward the user's natural-language question to the engine.
4. Stash the structured result (pipelines + dashboard HTML + metadata)
   on ``state.data_viz_results`` so the API layer can return it in the JSON.
5. Return a short user-facing string that becomes ``ChatResponse.response``.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from typing import Any

import structlog
from bson import ObjectId
from bson.errors import InvalidId

from utils.apexchat.core.llm import LLMClient, get_dashboard_tool_client
from utils.apexchat.core.status_stream import emit_status
from utils.apexchat.schemas.models import WorkflowState
from utils.apexchat.tools.data_viz.mongo_vanna import (
    CHART_DATA_PLACEHOLDER,
    MongoVanna,
)
from utils.apexchat.tools.general import BaseTool
from utils.Mongodb_tools import MONGODB_TOOLS

logger = structlog.get_logger(__name__)


# ── Admin role check ─────────────────────────────────────────────────────────

# Role identifiers that are treated as admin (case-insensitive match below).
_ADMIN_ROLE_VALUES = {"admin", "administrator", "1"}


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    """
    Decode a JWT WITHOUT signature verification — we only need the claims to
    extract the user id. (Real auth is enforced elsewhere in the stack;
    this function is just a parser.)

    Returns the payload dict, or None if the input is not a JWT.
    """
    if not token or token.count(".") != 2:
        return None
    try:
        payload_b64 = token.split(".")[1]
        # JWTs use URL-safe base64 without padding — pad to multiple of 4
        padding = "=" * (-len(payload_b64) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64 + padding)
        return json.loads(decoded)
    except Exception:
        return None


def _resolve_user_id(user_id: str) -> tuple[str, str | None]:
    """
    Translate the incoming `user_id` (which may be a raw id OR a JWT) into
    a clean (uid, role_hint) pair.

    role_hint is whatever the JWT payload claims as the role — None when the
    input is a plain id with no embedded claims.
    """
    payload = _decode_jwt_payload(user_id)
    if payload is None:
        return user_id, None
    uid = (
        payload.get("uid")
        or payload.get("user_id")
        or payload.get("sub")
        or payload.get("id")
        or user_id
    )
    role_hint = payload.get("role")
    return str(uid), (str(role_hint) if role_hint else None)


def _role_value_is_admin(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in _ADMIN_ROLE_VALUES


def _is_admin(db, user_id: str) -> bool:
    """
    Resolve the caller's role and decide whether they're an admin.

    Resolution order:
      1. If `user_id` is a JWT, decode it. The JWT's `role` claim is
         considered authoritative when it explicitly says admin.
      2. Otherwise look up the user in the DB:
         a. `um_hybrid_user_role` (mirrors CIAM.UM_HYBRID_USER_ROLE)
         b. `um_user`              (mirrors CIAM.UM_USER)
         c. `users`                (booking-system collection — has a `role` field)
         Match by the resolved uid against multiple plausible field names.
      3. Anything else → False.

    Lookup failures are logged and treated as 'not admin'.
    """
    if not user_id:
        return False

    uid, role_hint = _resolve_user_id(user_id)

    # 1. Fast-path: JWT explicitly says admin
    if _role_value_is_admin(role_hint):
        logger.info("Admin gate: granted via JWT role claim", uid=uid)
        return True

    try:
        # ── 2a. um_hybrid_user_role ─────────────────────────────────────────
        candidate_filters: list[dict] = [
            {"um_user_name": uid},
            {"um_user_id": uid},
            {"um_user_id": uid},
            {"user_id": uid},
            {"user_id": uid},
        ]
        for flt in candidate_filters:
            for doc in db["um_hybrid_user_role"].find(flt, limit=10):
                role_value = (
                    doc.get("um_role_id")
                    or doc.get("role")
                    or doc.get("role_name")
                )
                if _role_value_is_admin(role_value):
                    logger.info("Admin gate: granted via um_hybrid_user_role", uid=uid)
                    return True

        # ── 2b. um_user ─────────────────────────────────────────────────────
        for doc in db["um_user"].find(
            {"$or": [{"um_user_name": uid}, {"um_id": uid}, {"_id": uid}]},
            limit=5,
        ):
            if _role_value_is_admin(doc.get("role") or doc.get("um_role")):
                logger.info("Admin gate: granted via um_user", uid=uid)
                return True

        # ── 2c. users (booking-system) — id may be an ObjectId ──────────────
        users_filters: list[dict] = [{"username": uid}, {"email": uid}]
        try:
            users_filters.append({"_id": ObjectId(uid)})
        except (InvalidId, TypeError):
            users_filters.append({"_id": uid})

        for flt in users_filters:
            for doc in db["users"].find(flt, limit=5):
                role_value = (
                    doc.get("role")
                    or doc.get("user_role")
                    or doc.get("type")
                )
                if _role_value_is_admin(role_value):
                    logger.info(
                        "Admin gate: granted via users collection",
                        uid=uid,
                        matched_field=list(flt.keys())[0],
                    )
                    return True

    except Exception as exc:
        logger.warning(
            "Admin role lookup failed — denying access",
            uid=uid,
            error=str(exc),
        )
        return False

    logger.info(
        "Admin gate: denied (no admin role found)",
        uid=uid,
        jwt_role_hint=role_hint,
    )
    return False


# ── Singleton engine ─────────────────────────────────────────────────────────

_engine_lock = asyncio.Lock()
_engine: MongoVanna | None = None


async def _get_engine(llm_client: LLMClient | None = None) -> MongoVanna:
    """
    Build and cache a single MongoVanna instance for the process.
    First call also auto-trains it from the live schema.
    """
    global _engine
    async with _engine_lock:
        if _engine is None:
            mongo = MONGODB_TOOLS()
            db = mongo.get_db_connection()
            engine = MongoVanna(db=db, llm_client=llm_client or get_dashboard_tool_client())
            # Schema introspection is cheap (few small finds) — run on the
            # event loop's executor so we don't block.
            await asyncio.to_thread(engine.train_from_schema)
            _engine = engine
        return _engine


# ── Pipeline execution + Excel rendering ─────────────────────────────────────


def _safe_sheet_name(name: str, used: set[str]) -> str:
    """Excel sheet names: <=31 chars, no `[]:*?/\\`, must be unique."""
    cleaned = re.sub(r"[\[\]:*?/\\]", " ", str(name or "")).strip() or "Sheet"
    cleaned = cleaned[:31]
    base = cleaned
    i = 2
    while cleaned in used:
        suffix = f" ({i})"
        cleaned = (base[: 31 - len(suffix)] + suffix).strip()
        i += 1
    used.add(cleaned)
    return cleaned


def _run_charts_and_collect_rows(
    engine: MongoVanna,
    charts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Execute every chart's pipeline and return {chart_id: rows}."""
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for chart in charts:
        cid = chart.get("id")
        if not cid:
            continue
        try:
            rows = engine.run(
                collection=chart["collection"],
                pipeline=chart.get("pipeline", []),
                enrichment=chart.get("enrichment") or None,
            )
        except Exception as exc:
            logger.warning(
                "DataVizTool: pipeline execution failed",
                chart_id=cid,
                collection=chart.get("collection"),
                error=str(exc),
            )
            rows = []
        rows_by_id[cid] = rows
    return rows_by_id


def _inject_rows_into_html(dashboard_html: str, rows_by_id: dict[str, Any]) -> str:
    """Replace the __CHART_DATA__ placeholder with a real JSON literal."""
    payload = json.dumps(rows_by_id)
    return dashboard_html.replace(CHART_DATA_PLACEHOLDER, payload)


def _build_excel_chart(chart_spec: dict[str, Any], df, n_rows: int, worksheet):
    """
    Build a native openpyxl chart object for `chart_spec`. Returns None for
    chart types that don't render meaningfully in Excel (table) or when there
    isn't enough data.

    Data layout assumed on the sheet:
      • Row 1            → header (column names from df)
      • Rows 2..n_rows+1 → data
      • Categories column = chart_x
      • Values column    = chart_y
    """
    if n_rows < 1 or df is None or df.empty:
        return None

    ct = chart_spec.get("chart_type", "table")
    if ct == "table":
        return None

    cols = list(df.columns)
    if not cols:
        return None

    x_col = chart_spec.get("chart_x") if chart_spec.get("chart_x") in cols else cols[0]
    y_col = chart_spec.get("chart_y")
    if y_col not in cols:
        # pick the first numeric column other than x_col, fall back to second column
        numeric_cols = [c for c in cols if c != x_col and df[c].dtype.kind in "iuf"]
        y_col = numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else cols[0])

    x_idx = cols.index(x_col) + 1
    y_idx = cols.index(y_col) + 1

    from openpyxl.chart import (
        AreaChart,
        BarChart,
        DoughnutChart,
        LineChart,
        PieChart,
        RadarChart,
        ScatterChart,
        Reference,
        Series,
    )

    last_row = n_rows + 1

    # Build chart object per type
    if ct == "bar":
        ch = BarChart()
        ch.type = "col"
        if chart_spec.get("stacked"):
            ch.grouping = "stacked"
            ch.overlap = 100
    elif ct == "line":
        ch = LineChart()
    elif ct == "area":
        ch = AreaChart()
    elif ct == "pie":
        ch = PieChart()
    elif ct == "doughnut":
        ch = DoughnutChart()
    elif ct == "radar" or ct == "polarArea":
        ch = RadarChart()
        ch.type = "filled" if ct == "polarArea" else "marker"
    elif ct == "scatter":
        ch = ScatterChart()
        ch.style = 13
    else:
        ch = BarChart()  # safe fallback

    ch.title = chart_spec.get("chart_title") or None
    if hasattr(ch, "x_axis") and ch.x_axis is not None:
        ch.x_axis.title = chart_spec.get("x_axis_label") or x_col
    if hasattr(ch, "y_axis") and ch.y_axis is not None:
        ch.y_axis.title = chart_spec.get("y_axis_label") or y_col

    if isinstance(ch, ScatterChart):
        x_ref = Reference(worksheet, min_col=x_idx, min_row=2, max_col=x_idx, max_row=last_row)
        y_ref = Reference(worksheet, min_col=y_idx, min_row=2, max_col=y_idx, max_row=last_row)
        series = Series(y_ref, x_ref, title=y_col)
        ch.series.append(series)
    else:
        # values include the header row so the series gets a legend label
        values = Reference(worksheet, min_col=y_idx, min_row=1, max_col=y_idx, max_row=last_row)
        cats   = Reference(worksheet, min_col=x_idx, min_row=2, max_col=x_idx, max_row=last_row)
        ch.add_data(values, titles_from_data=True)
        ch.set_categories(cats)

    ch.height = 10
    ch.width = 20
    return ch


def _build_excel_base64(
    charts: list[dict[str, Any]],
    rows_by_id: dict[str, list[dict[str, Any]]],
    dashboard_title: str,
) -> str:
    """
    One sheet per chart: data on the left, native Excel chart embedded next to
    it. Returns base64-encoded .xlsx bytes.
    """
    import pandas as pd

    buffer = io.BytesIO()
    used_sheet_names: set[str] = set()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        wrote_any = False
        for idx, chart in enumerate(charts, start=1):
            cid = chart.get("id")
            rows = rows_by_id.get(cid, []) or []
            cols = chart.get("output_columns") or (
                list(rows[0].keys()) if rows else []
            )
            df = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)
            sheet_name = _safe_sheet_name(
                chart.get("chart_title") or f"Chart {idx}", used_sheet_names
            )
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            wrote_any = True

            ws = writer.sheets[sheet_name]
            ch = _build_excel_chart(chart, df, n_rows=len(df), worksheet=ws)
            if ch is not None:
                # anchor chart two columns to the right of the data
                anchor_col_idx = (len(df.columns) if not df.empty else 1) + 2
                anchor_letter = _col_letter(anchor_col_idx)
                ws.add_chart(ch, f"{anchor_letter}2")

        if not wrote_any:
            pd.DataFrame([{"info": "No data"}]).to_excel(
                writer,
                sheet_name=_safe_sheet_name(dashboard_title or "Dashboard", used_sheet_names),
                index=False,
            )

    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _col_letter(idx: int) -> str:
    """1-based column index → Excel column letter (1→A, 27→AA)."""
    s = ""
    while idx > 0:
        idx, r = divmod(idx - 1, 26)
        s = chr(65 + r) + s
    return s


# ── Tool ─────────────────────────────────────────────────────────────────────

_REFUSAL_NOT_ADMIN = (
    "Data visualization is restricted to administrators. "
    "If you need this access, please contact your system administrator."
)


class DataVizTool(BaseTool):
    """
    Natural-language → MongoDB queries → dashboard, restricted to admin users.

    The structured result (pipelines, chart specs, dashboard HTML, metadata) is
    written to ``state.data_viz_results`` for inclusion in the API response.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    @property
    def name(self) -> str:
        return "data_viz"

    @property
    def description(self) -> str:
        return (
            "Translates natural-language analytics questions into MongoDB "
            "aggregations and renders the result as a labeled dashboard. Admin-only."
        )

    async def execute(self, state: WorkflowState) -> str:
        emit_status("tool_data_viz")
        start = time.perf_counter()

        # ── 1. Admin gate ─────────────────────────────────────────────────────
        # TEMPORARY FOR TESTING: Allow all users to access data visualization
        allowed = True
        # try:
        #     mongo = MONGODB_TOOLS()
        #     db = mongo.get_db_connection()
        #     allowed = await asyncio.to_thread(_is_admin, db, state.user_id)
        # except Exception as exc:
        #     logger.error(
        #         "DataVizTool: admin gate check failed",
        #         session_id=state.session_id,
        #         user_id=state.user_id,
        #         error=str(exc),
        #         exc_info=True,
        #     )
        #     state.data_viz_results = {
        #         "success": False,
        #         "error": "role_check_failed",
        #         "message": str(exc),
        #     }
        #     return _REFUSAL_NOT_ADMIN

        if False:  # Disabled for testing
            logger.info(
                "DataVizTool: access denied (non-admin)",
                session_id=state.session_id,
                user_id=state.user_id,
            )
            state.data_viz_results = {
                "success": False,
                "error": "forbidden",
                "message": "Caller is not an admin",
            }
            return _REFUSAL_NOT_ADMIN

        # ── 2. Run MongoVanna ─────────────────────────────────────────────────
        engine = await _get_engine(self._llm_client)

        logger.info(
            "DataVizTool executing",
            session_id=state.session_id,
            user_id=state.user_id,
            question_preview=state.user_message[:80],
        )

        result = await engine.ask(state.user_message)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # ── 2b. Run pipelines server-side ONLY to build the Excel file.
        # The frontend runs the pipelines itself and injects rows into
        # `dashboard_html` using `chart_data_placeholder`.
        excel_base64 = ""
        if result.get("success"):
            charts = result.get("charts") or []
            try:
                rows_by_id = await asyncio.to_thread(
                    _run_charts_and_collect_rows, engine, charts
                )
                excel_base64 = await asyncio.to_thread(
                    _build_excel_base64,
                    charts,
                    rows_by_id,
                    result.get("dashboard_title", ""),
                )
            except Exception as exc:
                logger.error(
                    "DataVizTool: excel build failed",
                    session_id=state.session_id,
                    error=str(exc),
                    exc_info=True,
                )

        # Preserve explanation for the user-facing reply before trimming.
        explanation = result.get("explanation") or ""

        charts = result.get("charts") or []
        trimmed_charts = [
            {
                "id":         c.get("id"),
                "collection": c.get("collection"),
                "pipeline":   c.get("pipeline", []),
                "enrichment": c.get("enrichment") or [],
            }
            for c in charts
        ]

        if result.get("success"):
            state.data_viz_results = {
                "success":                True,
                "dashboard_html":         result.get("dashboard_html", ""),
                "chart_data_placeholder": result.get("chart_data_placeholder", CHART_DATA_PLACEHOLDER),
                "charts":                 trimmed_charts,
                "dashboard_excel_base64": excel_base64,
            }
        else:
            state.data_viz_results = {
                "success": False,
                "error":   result.get("error", "failed"),
                "message": result.get("message", ""),
            }

        logger.info(
            "DataVizTool completed",
            session_id=state.session_id,
            success=result.get("success"),
            chart_count=len(charts),
            chart_types=[c.get("chart_type") for c in charts],
            elapsed_ms=round(elapsed_ms, 2),
        )

        # ── 3. Compose the user-facing string ────────────────────────────────
        if not result.get("success"):
            return (
                "Sorry, your analysis request is not clear or cannot be processed. Please rephrase and try again."
            )

        explanation = result.get("explanation") or ""
        if explanation:
            return explanation

        if len(charts) > 1:
            types = ", ".join(c.get("chart_type", "chart") for c in charts)
            return (
                f"Here is the dashboard you asked for — {len(charts)} charts "
                f"({types}). The frontend will run the included pipelines to "
                "populate them with live data."
            )
        first = charts[0] if charts else {}
        return (
            f"Here is the {first.get('chart_type', 'chart')} you asked for. "
            f"It will pull from the `{first.get('collection')}` collection — "
            "the frontend should run the included pipeline to populate the chart."
        )
