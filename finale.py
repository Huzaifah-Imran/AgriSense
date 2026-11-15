import flet as ft
import os, math, asyncio, random, time
import requests, pandas as pd

from crewai import Agent, Task, Process, Crew, LLM
from crewai.tools import tool
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACKING"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"


# ============================================================
#  DATE HELPERS
# ============================================================
def _normalize_date(value, default_days_ago: int) -> str:
    if value is None:
        return (datetime.now() - timedelta(days=default_days_ago)).strftime("%Y%m%d")
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "today", "now", "current"):
        return (datetime.now() - timedelta(days=default_days_ago)).strftime("%Y%m%d")
    if len(s) == 8 and s.isdigit():
        return s
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
                "%Y.%m.%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return (datetime.now() - timedelta(days=default_days_ago)).strftime("%Y%m%d")


# ============================================================
#  TOOLS
# ============================================================
@tool("Get Coordinates Tool")
def get_coordinates(address: str) -> str:
    """Retrieves the latitude and longitude of a location.

    Pass a clean place name (e.g. "Thar, Sindh" or "Lahore, Pakistan").
    """
    geolocator = Nominatim(user_agent="agrisense_app")
    try:
        loc = geolocator.geocode(address, timeout=10)
        if loc:
            return (f"Location: {address}\n"
                    f"Latitude: {loc.latitude}\n"
                    f"Longitude: {loc.longitude}")
        return f"Location '{address}' not found."
    except Exception as e:
        return f"Error: {e}"


@tool("fetch_nasa_power_agricultural_data")
def fetch_nasa_power_agricultural_data(
    latitude: float,
    longitude: float,
    start_date: str = "",
    end_date: str = "",
) -> str:
    """Fetch NASA POWER daily agricultural climate data for a location."""
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return ("ERROR: latitude and longitude must be numeric. "
                "Call the Get Coordinates Tool first to obtain them.")

    end_date = _normalize_date(end_date, 0)
    start_date = _normalize_date(start_date, 30)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    parameters = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR",
                  "ALLSKY_SFC_SW_DWN", "WS2M", "T2MDEW",
                  "GWETTOP", "GWETROOT", "GWETPROF"]
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(parameters), "community": "AG",
        "longitude": longitude, "latitude": latitude,
        "start": start_date, "end": end_date,
        "format": "JSON", "time-standard": "UTC",
    }

    try:
        r = requests.get(url, params=params, timeout=300)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("properties", {}).get("parameter", {})
        if not data:
            return (f"NASA POWER returned no data for lat={latitude}, "
                    f"lon={longitude}, {start_date}..{end_date}.")
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "DATE"
        df = df.replace(-999, float("nan"))
        return df.to_csv()
    except requests.exceptions.RequestException as e:
        return f"NASA POWER request failed ({e})."
    except (KeyError, ValueError) as e:
        return f"NASA POWER processing failed: {e}"


# ============================================================
#  STEP FORMATTER (fallback for very old CrewAI builds only)
# ============================================================
def format_step(step):
    try:
        if hasattr(step, "tool") and hasattr(step, "tool_input"):
            tool = getattr(step, "tool", "tool")
            ti = getattr(step, "tool_input", {})
            if isinstance(ti, dict):
                input_str = ", ".join(
                    f"{k}={str(v)[:60]}" for k, v in ti.items()
                )
            else:
                input_str = str(ti)[:120]
            thought = str(getattr(step, "thought", "") or "").strip().replace("\n", " ")
            if len(thought) > 150:
                thought = thought[:150] + "…"
            if thought:
                return "#3B82F6", f"🤔 {thought}\n🔧 {tool}({input_str})"
            return "#3B82F6", f"🔧 Calling {tool}({input_str})"

        if hasattr(step, "return_values"):
            rv = getattr(step, "return_values", None)
            out = rv.get("output", "") if isinstance(rv, dict) else str(rv)
            return "#22C55E", f"✅ Final answer: {str(out)[:200]}"

        log = getattr(step, "log", None)
        if log:
            return "#888888", str(log).replace("\n", " ")[:220]

        if isinstance(step, dict):
            if "tool" in step:
                return "#3B82F6", f"🔧 {step.get('tool')}({step.get('tool_input', '')})"
            if "output" in step:
                return "#22C55E", f"✅ {str(step['output'])[:200]}"

        s = str(step).strip().replace("\n", " ")
        if not s:
            return "#888888", ""
        return "#888888", s[:200] + ("…" if len(s) > 200 else "")
    except Exception:
        return "#888888", ""


# ============================================================
#  CREWAI EVENT BUS  ->  Live Activity feed
# ============================================================
_activity_sink = None
_EVENT_BUS = None
_EVENTS = {}
_LISTENERS_DONE = False


def _emit(text, color="#888888"):
    sink = _activity_sink
    if sink is None or not text:
        return
    try:
        sink(text, color)
    except Exception:
        pass


def _short(value, n=180):
    s = str(value).replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"


def _fmt_args(args, max_len=140):
    if isinstance(args, dict):
        s = ", ".join(f"{k}={_short(v, 60)}" for k, v in args.items())
    else:
        s = _short(args, max_len)
    return s if len(s) <= max_len else s[:max_len] + "…"


def _load_event_bus():
    global _EVENT_BUS, _EVENTS
    for mod_name in ("crewai.events", "crewai.utilities.events"):
        try:
            mod = __import__(mod_name, fromlist=["crewai_event_bus"])
        except Exception:
            continue
        bus = getattr(mod, "crewai_event_bus", None)
        if bus is None:
            continue
        _EVENT_BUS = bus
        for n in ("ToolUsageStartedEvent", "ToolUsageFinishedEvent",
                  "ToolUsageErrorEvent", "ToolExecutionErrorEvent",
                  "AgentExecutionStartedEvent", "AgentExecutionCompletedEvent",
                  "TaskStartedEvent", "TaskCompletedEvent",
                  "CrewKickoffStartedEvent", "CrewKickoffCompletedEvent",
                  "LLMCallStartedEvent", "LLMCallCompletedEvent"):
            cls = getattr(mod, n, None)
            if cls is not None:
                _EVENTS[n] = cls
        if "ToolUsageStartedEvent" not in _EVENTS:
            try:
                tm = __import__(f"{mod_name}.tool_usage_events",
                                fromlist=["ToolUsageStartedEvent"])
                for n in ("ToolUsageStartedEvent", "ToolUsageFinishedEvent",
                          "ToolUsageErrorEvent", "ToolExecutionErrorEvent"):
                    cls = getattr(tm, n, None)
                    if cls is not None:
                        _EVENTS[n] = cls
            except Exception:
                pass
        if _EVENTS:
            return True
    return False


def _register_listeners():
    global _LISTENERS_DONE
    if _LISTENERS_DONE or _EVENT_BUS is None or not _EVENTS:
        return
    _LISTENERS_DONE = True

    def _reg(name, handler):
        cls = _EVENTS.get(name)
        if cls is None:
            return
        def _safe(source, event):
            try:
                handler(source, event)
            except Exception:
                pass
        try:
            _EVENT_BUS.on(cls)(_safe)
        except Exception:
            pass

    def on_crew_started(source, event):
        _emit(f"🚀 Crew started: {_short(getattr(event, 'crew_name', 'crew'), 60)}", "#F59E0B")

    def on_crew_done(source, event):
        _emit("🎉 Crew finished", "#22C55E")

    def on_task_started(source, event):
        task = getattr(event, "task", None)
        desc = getattr(task, "description", "") if task is not None else ""
        if not desc:
            desc = getattr(event, "task_description", "") or "task"
        _emit(f"📋 Task started: {_short(desc, 140)}", "#F59E0B")

    def on_task_done(source, event):
        _emit("✅ Task completed", "#22C55E")

    def on_agent_started(source, event):
        agent = getattr(event, "agent", None)
        role = getattr(agent, "role", None) if agent is not None else None
        _emit(f"👤 Agent started: {role or 'agent'}", "#F59E0B")

    def on_agent_done(source, event):
        agent = getattr(event, "agent", None)
        role = getattr(agent, "role", None) if agent is not None else None
        out = getattr(event, "output", "") or ""
        _emit(f"✅ {role or 'Agent'} finished: {_short(out, 160)}", "#22C55E")

    def on_tool_started(source, event):
        name = getattr(event, "tool_name", None) or getattr(event, "tool", "tool")
        args = getattr(event, "tool_args", None)
        if args is None:
            args = getattr(event, "tool_input", {})
        _emit(f"🔧 Calling {name}({_fmt_args(args)})", "#3B82F6")

    def on_tool_finished(source, event):
        name = getattr(event, "tool_name", None) or getattr(event, "tool", "tool")
        out = getattr(event, "output", "") or getattr(event, "result", "")
        _emit(f"📥 {name} returned: {_short(out, 160)}", "#22C55E")

    def on_tool_error(source, event):
        name = getattr(event, "tool_name", None) or getattr(event, "tool", "tool")
        err = getattr(event, "error", None) or getattr(event, "exception", "") or "unknown error"
        _emit(f"❌ Tool error in {name}: {_short(err, 160)}", "#DC2626")

    def on_llm_started(source, event):
        _emit("🧠 Thinking…", "#8B5CF6")

    _reg("CrewKickoffStartedEvent", on_crew_started)
    _reg("CrewKickoffCompletedEvent", on_crew_done)
    _reg("TaskStartedEvent", on_task_started)
    _reg("TaskCompletedEvent", on_task_done)
    _reg("AgentExecutionStartedEvent", on_agent_started)
    _reg("AgentExecutionCompletedEvent", on_agent_done)
    _reg("ToolUsageStartedEvent", on_tool_started)
    _reg("ToolUsageFinishedEvent", on_tool_finished)
    _reg("ToolUsageErrorEvent", on_tool_error)
    _reg("ToolExecutionErrorEvent", on_tool_error)
    _reg("LLMCallStartedEvent", on_llm_started)


_load_event_bus()
_register_listeners()


def _make_step_callback():
    if "ToolUsageStartedEvent" in _EVENTS or "ToolUsageFinishedEvent" in _EVENTS:
        return None
    def _cb(step):
        color, text = format_step(step)
        if text:
            _emit(text, color)
    return _cb


# ============================================================
#  CREW BUILDERS
# ============================================================
def build_crop_crew(model, location, crop, days=30):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    location_agent = Agent(
        role="Location Finder",
        goal="Finds exact coordinates of given locations.",
        backstory="Specialist in geocoding.",
        tools=[get_coordinates], llm=model, verbose=False,
    )
    weather_agent = Agent(
        role="Climate Data Analyst",
        goal="Fetches and analyzes NASA climate data.",
        backstory="Expert in NASA environmental metrics.",
        tools=[fetch_nasa_power_agricultural_data],
        llm=model, inject_date=True, date_format="%Y%m%d", verbose=False,
    )
    task = Task(
        description=(
            f"The user lives in {location} and wants to grow {crop}. "
            "Determine if it is suitable.\n\n"
            f"Analysis window: {start_str} → {end_str} ({days} days).\n\n"
            "1. Use Location Finder for coordinates.\n"
            "2. Use Climate Data Analyst to call "
            f"'fetch_nasa_power_agricultural_data' with start_date={start_str} "
            f"and end_date={end_str}.\n"
            "3. Analyze the climate conditions. If the dataset is large, "
            "   summarize into averages, min/max, totals, and trends instead "
            "   of listing every row.\n"
            f"4. Decide if {crop} is suitable or recommend alternatives.\n"
            "5. When reporting the location, use ONLY the location name "
            f"({location}). Do not paste full street/postal addresses.\n"
        ),
        expected_output=(
            "A recommendation with:\n"
            "1. Location name and coordinates (short form).\n"
            "2. NASA climate summary (averages / totals / trends).\n"
            "3. Climate analysis.\n"
            f"4. {crop} suitability verdict for a {days}-day window.\n"
            "5. Alternative crops if needed."
        ),
    )
    manager_agent = Agent(
        role="Agricultural Research Manager",
        goal="Orchestrate location discovery and climate analysis.",
        backstory="Expert agricultural research manager.",
        llm=model, verbose=False,
    )
    return Crew(
        agents=[location_agent, weather_agent], tasks=[task],
        process=Process.hierarchical, manager_llm=model,
        manager_agent=manager_agent, verbose=True,
        step_callback=_make_step_callback(),
    )


def build_query_crew(model, query):
    general_agent = Agent(
        role="Agricultural AI Assistant",
        goal=(
            "Answer any farming, climate, or agriculture question. "
            "Use the available tools whenever location or climate data is "
            "needed to give a data-driven answer."
        ),
        backstory=(
            "An expert agricultural scientist with access to a geocoding "
            "tool and the NASA POWER climate dataset."
        ),
        tools=[get_coordinates, fetch_nasa_power_agricultural_data],
        llm=model,
        inject_date=True,
        date_format="%Y%m%d",
        verbose=False,
    )
    task = Task(
        description=(
            f"User question: {query}\n\n"
            "Follow these rules:\n"
            "1. If the question mentions or implies a location, first call "
            "   the 'Get Coordinates Tool' to obtain its latitude and "
            "   longitude. Do NOT invent coordinates.\n"
            "2. Then call 'fetch_nasa_power_agricultural_data' with those "
            "   coordinates. Leave start_date/end_date empty to get the "
            "   last 30 days automatically, or pass YYYYMMDD dates.\n"
            "3. Base your answer on the returned climate data.\n"
            "4. If the question truly needs no location or climate data, "
            "   answer directly without tools.\n"
            "5. When mentioning the location, use only the place name "
            "   (e.g. 'Thar, Sindh'). Do not paste full postal addresses.\n"
        ),
        expected_output=(
            "A clear, well-structured answer that directly addresses the "
            "user's question. If NASA climate data was used, cite the key "
            "figures that support your conclusion. Keep the location name "
            "short — no full street/postal addresses."
        ),
        agent=general_agent,
    )
    return Crew(
        agents=[general_agent], tasks=[task],
        process=Process.sequential, verbose=True,
        step_callback=_make_step_callback(),
    )


# ============================================================
#  UI
# ============================================================
def main(page: ft.Page):
    page.title = "AgriSense — AI Crop Suitability"
    page.bgcolor = "#F7F7F8"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.spacing = 0
    try:
        page.window.width = 1400
        page.window.height = 880
    except Exception:
        pass

    state = {
        "model_type": "google",
        "google_key": "",
        "google_model": "gemma-4-27b-it",
        "ollama_url": "",
        "ollama_model": "gemma4",
        "running": False,
        "result": None,
        "error": None,
    }

    # -------- dialog helpers --------
    def open_dialog(dialog):
        try:
            page.show_dialog(dialog)
        except AttributeError:
            try:
                page.open(dialog)
            except AttributeError:
                page.overlay.append(dialog)
                dialog.open = True
                page.update()

    def close_dialog(dialog):
        try:
            page.pop_dialog()
        except AttributeError:
            try:
                page.close(dialog)
            except AttributeError:
                dialog.open = False
                page.update()

    # -------- visualizer --------
    ring_outer = ft.Container(
        width=420, height=420,
        border=ft.Border.all(1, "#DADADD"), border_radius=1000,
        animate=ft.Animation(120, ft.AnimationCurve.EASE_OUT),
    )
    ring_mid = ft.Container(
        width=300, height=300,
        border=ft.Border.all(1, "#E5E5E8"), border_radius=1000,
        animate=ft.Animation(120, ft.AnimationCurve.EASE_OUT),
    )
    ring_inner = ft.Container(
        width=180, height=180,
        border=ft.Border.all(1, "#EFEFF2"), border_radius=1000,
        animate=ft.Animation(120, ft.AnimationCurve.EASE_OUT),
    )
    ring_center = ft.Container(
        width=88, height=88, bgcolor="#111",
        border_radius=1000, alignment=ft.Alignment(0, 0),
        content=ft.Icon(ft.Icons.GRASS, color="white", size=40),
        shadow=ft.BoxShadow(blur_radius=24, spread_radius=2, color="#22000000"),
    )
    visualizer = ft.Stack(
        [ring_outer, ring_mid, ring_inner, ring_center],
        alignment=ft.Alignment(0, 0), width=520, height=520,
    )
    status_title = ft.Text(
        "Ready to Analyze", size=17,
        weight=ft.FontWeight.W_600, color="#111",
        text_align=ft.TextAlign.CENTER,
    )
    status_sub = ft.Text(
        "Enter a location and crop, or ask a custom question",
        size=12, color="#888", text_align=ft.TextAlign.CENTER,
    )
    visualizer_column = ft.Column(
        [visualizer, ft.Container(height=4), status_title, status_sub],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        alignment=ft.MainAxisAlignment.CENTER, spacing=6, expand=True,
    )

    # -------- live activity feed --------
    activity_list = ft.ListView(
        expand=True, spacing=8, auto_scroll=True,
        padding=ft.Padding.symmetric(vertical=4),
    )
    activity_status = ft.Text("Running...", size=11, color="#888")

    def add_activity(text, icon_color="#22C55E"):
        if len(activity_list.controls) > 250:
            activity_list.controls = activity_list.controls[-250:]
        ts = datetime.now().strftime("%H:%M:%S")
        item = ft.Row(
            [
                ft.Container(
                    width=6, height=6, bgcolor=icon_color,
                    border_radius=10, margin=ft.Margin.only(top=6),
                ),
                ft.Column([
                    ft.Text(ts, size=9, color="#AAA",
                            weight=ft.FontWeight.W_500),
                    ft.Text(text, size=11, color="#333",
                            selectable=True, expand=True),
                ], spacing=0, expand=True),
            ],
            spacing=8, vertical_alignment=ft.CrossAxisAlignment.START,
        )
        activity_list.controls.append(item)
        try:
            page.update()
        except Exception:
            pass

    activity_card = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.TIMELINE, size=16, color="#111"),
                ft.Text("Live Activity", size=13,
                        weight=ft.FontWeight.W_600, color="#111"),
                ft.Container(expand=True),
                activity_status,
            ]),
            ft.Divider(height=1, color="#EAEAEC"),
            ft.Container(content=activity_list, expand=True, height=220),
        ], spacing=8),
        bgcolor="#FFFFFF", border_radius=16,
        border=ft.Border.all(1, "#EAEAEC"), padding=18,
        visible=False, expand=True,
    )

    # -------- result / error cards --------
    try:
        result_md = ft.Markdown(
            "", selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        )
    except Exception:
        result_md = ft.Markdown("", selectable=True)

    result_card = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.ANALYTICS_OUTLINED, size=20, color="#111"),
                ft.Text("Analysis Report", size=16,
                        weight=ft.FontWeight.W_700, color="#111"),
            ], spacing=8),
            ft.Divider(height=1, color="#EAEAEC"),
            ft.Container(
                content=ft.Column([result_md], scroll=ft.ScrollMode.AUTO),
                expand=True,
                padding=ft.Padding.only(top=8),
            ),
        ], spacing=12),
        bgcolor="#FFFFFF", border_radius=16,
        border=ft.Border.all(1, "#EAEAEC"), padding=24,
        visible=False, expand=True,
    )

    error_text = ft.Text("", size=12, color="#666",
                         text_align=ft.TextAlign.CENTER)
    error_card = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Icon(ft.Icons.ERROR_OUTLINE, size=32, color="#DC2626"),
                width=64, height=64, bgcolor="#FEE2E2",
                border_radius=1000, alignment=ft.Alignment(0, 0),
            ),
            ft.Text("Analysis Failed", size=16,
                    weight=ft.FontWeight.W_600, color="#111"),
            error_text,
        ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER, spacing=10),
        alignment=ft.Alignment(0, 0), visible=False, expand=True, padding=24,
    )

    view_column = ft.Column(
        [visualizer_column, activity_card, result_card, error_card],
        expand=True, spacing=16, scroll=ft.ScrollMode.AUTO,
    )
    view_holder = ft.Container(content=view_column, expand=True)

    def reset_rings():
        ring_outer.width, ring_outer.height = 420, 420
        ring_mid.width, ring_mid.height = 300, 300
        ring_inner.width, ring_inner.height = 180, 180

    def show_idle():
        reset_rings()
        visualizer_column.visible = True
        activity_card.visible = False
        result_card.visible = False
        error_card.visible = False
        status_title.value = "Ready to Analyze"
        status_sub.value = "Enter a location and crop, or ask a custom question"
        page.update()

    def show_busy(title, sub):
        visualizer_column.visible = True
        activity_card.visible = True
        result_card.visible = False
        error_card.visible = False
        status_title.value = title
        status_sub.value = sub
        activity_status.value = "Running..."
        activity_status.color = "#888"
        page.update()

    def show_result(md):
        result_md.value = md
        visualizer_column.visible = False
        activity_card.visible = False
        result_card.visible = True
        error_card.visible = False
        page.update()

    def show_error(msg):
        error_text.value = msg
        visualizer_column.visible = False
        activity_card.visible = False
        result_card.visible = False
        error_card.visible = True
        page.update()

    # -------- model badge --------
    def badge_text():
        if state["model_type"] == "google":
            return f"Google · {state['google_model']}"
        return f"Ollama · {state['ollama_model']}"

    badge_label = ft.Text(badge_text(), size=11,
                          color="#444", weight=ft.FontWeight.W_500)
    model_badge = ft.Container(
        content=ft.Row([
            ft.Container(width=6, height=6, bgcolor="#22C55E", border_radius=10),
            badge_label,
        ], spacing=6),
        padding=ft.Padding.symmetric(horizontal=10, vertical=6),
        bgcolor="#F0F0F2", border_radius=20,
    )

    def update_badge():
        badge_label.value = badge_text()
        page.update()

    # -------- settings dialog --------
    def open_settings(e=None):
        model_type_radio = ft.RadioGroup(
            value=state["model_type"],
            content=ft.Row([
                ft.Radio(value="google", label="Google AI Studio"),
                ft.Radio(value="ollama", label="Ollama (Pinggy)"),
            ], spacing=20),
        )
        google_key_field = ft.TextField(
            label="Google API Key", password=True, can_reveal_password=True,
            value=state["google_key"], border_radius=12, dense=True,
        )
        google_model_field = ft.TextField(
            label="Google Model", value=state["google_model"],
            border_radius=12, dense=True,
        )
        ollama_url_field = ft.TextField(
            label="Pinggy URL", value=state["ollama_url"], border_radius=12,
            dense=True, hint_text="https://xxxx.run.pinggy-free.link",
        )
        ollama_model_field = ft.TextField(
            label="Ollama Model", value=state["ollama_model"],
            border_radius=12, dense=True,
        )
        google_section = ft.Column([google_key_field, google_model_field], spacing=12)
        ollama_section = ft.Column([
            ft.Text("Expose Ollama running on Kaggle via Pinggy",
                    size=11, color="#888"),
            ollama_url_field, ollama_model_field,
        ], spacing=10, visible=(state["model_type"] == "ollama"))

        def on_type_change(e):
            is_google = model_type_radio.value == "google"
            google_section.visible = is_google
            ollama_section.visible = not is_google
            page.update()
        model_type_radio.on_change = on_type_change

        def save(e):
            state["model_type"] = model_type_radio.value
            state["google_key"] = google_key_field.value or ""
            state["google_model"] = google_model_field.value or "gemma-4-27b-it"
            state["ollama_url"] = ollama_url_field.value or ""
            state["ollama_model"] = ollama_model_field.value or "gemma4"
            update_badge()
            close_dialog(dialog)

        dialog = ft.AlertDialog(
            title=ft.Text("Model Settings", weight=ft.FontWeight.W_600),
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Choose your AI backend", size=12, color="#666"),
                    model_type_radio,
                    ft.Divider(height=1, color="#EAEAEC"),
                    google_section, ollama_section,
                ], spacing=14, tight=True),
                width=420,
            ),
            actions=[
                ft.TextButton(
                    "Cancel",
                    on_click=lambda e: close_dialog(dialog),
                ),
                ft.TextButton(
                    "Save", on_click=save,
                    style=ft.ButtonStyle(
                        bgcolor="#111", color="white",
                        padding=ft.Padding.symmetric(horizontal=20, vertical=10),
                        shape=ft.RoundedRectangleBorder(radius=8),
                    ),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        open_dialog(dialog)

    # -------- animation --------
    cancel_token = {"id": None}

    async def animation_task(token):
        t = 0.0
        while state["running"] and cancel_token["id"] == token:
            t += 0.1
            pulse = (math.sin(t) + 1) / 2
            base = 20 + pulse * 45
            ring_outer.width = 420 + base
            ring_outer.height = 420 + base
            ring_mid.width = 300 + base * 0.65
            ring_mid.height = 300 + base * 0.65
            ring_inner.width = 180 + base * 0.35
            ring_inner.height = 180 + base * 0.35
            try:
                page.update()
            except Exception:
                pass
            await asyncio.sleep(0.05)

        if cancel_token["id"] != token:
            return
        await asyncio.sleep(0.2)
        if cancel_token["id"] != token:
            return

        if state["error"]:
            show_error(state["error"])
        elif state["result"]:
            show_result(state["result"])
        else:
            show_idle()

    # -------- crew worker --------
    def crew_worker(mode_val, loc, crop, query, days):
        global _activity_sink
        try:
            if state["model_type"] == "google":
                if not state["google_key"]:
                    raise ValueError("Google API key is not set.")
                model = LLM(model=f"google/{state['google_model']}",
                            api_key=state["google_key"], timeout=600)
            else:
                if not state["ollama_url"]:
                    raise ValueError("Pinggy / Ollama URL is not set.")
                model = LLM(model=f"ollama/{state['ollama_model']}",
                            base_url=state["ollama_url"], timeout=600)

            if mode_val == "crop":
                add_activity(
                    f"📍 Analysing whether '{crop}' suits {loc} "
                    f"over the last {days} days",
                    icon_color="#3B82F6",
                )
                crew = build_crop_crew(model, loc, crop, days)
            else:
                preview = query[:120] + ("…" if len(query) > 120 else "")
                add_activity(f"💬 Answering: {preview}",
                             icon_color="#3B82F6")
                crew = build_query_crew(model, query)

            _activity_sink = lambda text, color="#888888": add_activity(
                text, icon_color=color
            )
            _t0 = time.time()
            try:
                result = crew.kickoff()
            finally:
                _activity_sink = None
            elapsed = time.time() - _t0

            print(result)

            state["result"] = str(result)
            add_activity(f"🎉 Done in {elapsed:.1f}s", icon_color="#22C55E")

        except Exception as ex:
            _activity_sink = None
            state["error"] = f"{type(ex).__name__}: {ex}"
            add_activity(f"❌ Error: {state['error']}",
                         icon_color="#DC2626")
        finally:
            state["running"] = False

    # -------- trigger --------
    def on_analyze(e):
        if state["running"]:
            return

        if mode_radio.value == "crop":
            loc = (location_field.value or "").strip()
            crop = (crop_field.value or "").strip()
            if not loc or not crop:
                show_error("Please provide both location and crop.")
                return
            days = int(duration_slider.value)
            busy_title = f"Analyzing {crop} in {loc}…"
            busy_sub = (f"Gathering coordinates and {days} days "
                        "of NASA climate data")
            args = ("crop", loc, crop, "", days)
        else:
            query = (query_field.value or "").strip()
            if not query:
                show_error("Please enter a question.")
                return
            busy_title = "Thinking…"
            busy_sub = "The AI agent is working on your question"
            args = ("query", "", "", query, 30)

        if state["model_type"] == "google" and not state["google_key"]:
            open_settings()
            return
        if state["model_type"] == "ollama" and not state["ollama_url"]:
            open_settings()
            return

        activity_list.controls.clear()

        state["running"] = True
        state["result"] = None
        state["error"] = None
        token = random.random()
        cancel_token["id"] = token

        show_busy(busy_title, busy_sub)
        page.run_task(animation_task, token)
        page.run_thread(crew_worker, *args)

    # -------- inputs --------
    location_field = ft.TextField(
        label="Location", value="Thar, Sindh",
        prefix_icon=ft.Icons.LOCATION_ON_OUTLINED,
        border_radius=12, filled=True, bgcolor="#F7F7F8",
        border_color="transparent",
    )
    crop_field = ft.TextField(
        label="Crop", value="rice",
        prefix_icon=ft.Icons.GRASS,
        border_radius=12, filled=True, bgcolor="#F7F7F8",
        border_color="transparent",
    )
    query_field = ft.TextField(
        label="Ask anything",
        hint_text="e.g. What crops grow well in Thar with limited water?",
        multiline=True, min_lines=5, max_lines=10,
        border_radius=12, filled=True, bgcolor="#F7F7F8",
        border_color="transparent",
    )

    # -------- duration slider (crop mode only) --------
    duration_value_text = ft.Text("Last 30 days", size=11, color="#888",
                                   weight=ft.FontWeight.W_500)
    duration_slider = ft.Slider(
        min=1, max=365, divisions=364, value=30,
        active_color="#111", inactive_color="#E5E5E8",
        label="{value} days",
    )

    def on_duration_change(e):
        v = int(duration_slider.value)
        duration_value_text.value = (
            f"Last {v} day" if v == 1 else f"Last {v} days"
        )
        page.update()
    duration_slider.on_change = on_duration_change

    duration_block = ft.Column([
        ft.Row([
            ft.Text("Analysis Duration", size=11, color="#666",
                    weight=ft.FontWeight.W_500),
            ft.Container(expand=True),
            duration_value_text,
        ]),
        duration_slider,
    ], spacing=0)

    mode_radio = ft.RadioGroup(
        value="crop",
        content=ft.Row([
            ft.Radio(value="crop", label="Crop Analysis"),
            ft.Radio(value="query", label="Custom Query"),
        ], spacing=16),
    )

    crop_inputs = ft.Column(
        [location_field, crop_field, duration_block],
        spacing=12,
    )
    query_inputs = ft.Column([query_field], spacing=12, visible=False)

    def on_mode_change(e):
        is_crop = mode_radio.value == "crop"
        crop_inputs.visible = is_crop
        query_inputs.visible = not is_crop
        page.update()

    mode_radio.on_change = on_mode_change

    analyze_btn = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.AUTO_AWESOME, color="white", size=18),
            ft.Text("Run Analysis", color="white", size=14,
                    weight=ft.FontWeight.W_600),
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=8),
        padding=ft.Padding.symmetric(vertical=16),
        bgcolor="#111", border_radius=12,
        on_click=on_analyze, ink=True,
    )

    # -------- header --------
    header = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Container(
                    content=ft.Icon(ft.Icons.ECO, color="white", size=22),
                    width=40, height=40, bgcolor="#111",
                    border_radius=12, alignment=ft.Alignment(0, 0),
                ),
                ft.Column([
                    ft.Text("AgriSense", size=18,
                            weight=ft.FontWeight.W_700, color="#111"),
                    ft.Text("AI crop suitability analyzer",
                            size=11, color="#888"),
                ], spacing=0),
            ], spacing=12),
            ft.Container(expand=True),
            ft.Row([
                model_badge,
                ft.IconButton(
                    icon=ft.Icons.SETTINGS_OUTLINED, icon_color="#111",
                    tooltip="Model settings", on_click=open_settings,
                ),
            ], spacing=4),
        ]),
        padding=ft.Padding.symmetric(horizontal=28, vertical=16),
        bgcolor="#FFFFFF",
        border=ft.Border.only(bottom=ft.BorderSide(1, "#EAEAEC")),
    )

    # -------- left panel --------
    left_panel = ft.Container(
        content=ft.Column([
            ft.Text("Configure Analysis", size=15,
                    weight=ft.FontWeight.W_600, color="#111"),
            ft.Text("Pick a mode and fill in the details.",
                    size=12, color="#777"),
            ft.Container(height=4),
            mode_radio,
            ft.Divider(height=1, color="#EAEAEC"),
            crop_inputs,
            query_inputs,
            ft.Container(height=4),
            analyze_btn,
            ft.Container(expand=True),
            ft.Row([
                ft.Icon(ft.Icons.INFO_OUTLINE, size=14, color="#888"),
                ft.Text("Powered by CrewAI + NASA POWER",
                        size=10, color="#888"),
            ], spacing=6),
        ], spacing=12, scroll=ft.ScrollMode.AUTO),
        padding=24, bgcolor="#FFFFFF",
        border_radius=16, border=ft.Border.all(1, "#EAEAEC"),
        width=380,
    )

    body = ft.Container(
        content=ft.Row([
            left_panel,
            ft.Container(content=view_holder, expand=True),
        ], spacing=20, vertical_alignment=ft.CrossAxisAlignment.START),
        padding=24, expand=True,
    )

    page.add(ft.Column([header, body], spacing=0, expand=True))
    show_idle()


if __name__ == "__main__":
    ft.run(main)