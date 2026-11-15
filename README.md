# 🌱 AgriSense — AI Crop Suitability Analyzer

AgriSense is a desktop app that tells you whether a crop will grow well in a given location. It combines **real NASA climate data** with a **multi-agent AI system** ([CrewAI](https://www.crewai.com/)) and presents everything in a clean, modern [Flet](https://flet.dev/) interface with a live view of what the agents are doing.

You can either run a structured **Crop Analysis** (location + crop + time window) or ask any **Custom Query** about farming, climate, or agriculture.

---

## ✨ Features

- **Two modes**
  - **Crop Analysis** — enter a location (e.g. `Thar, Sindh`), a crop (e.g. `rice`), and an analysis window (1–365 days). You get a climate summary, a suitability verdict, and alternative crop suggestions.
  - **Custom Query** — ask anything, e.g. *"What crops grow well in Thar with limited water?"*. The agent decides whether it needs location and climate data.
- **Real climate data** — daily NASA POWER agricultural data: temperature (mean/max/min), relative humidity, precipitation, solar radiation, wind speed, dew point, and soil wetness (surface, root zone, profile).
- **Automatic geocoding** — place names are converted to coordinates with OpenStreetMap Nominatim.
- **Live Activity feed** — watch agents, tasks, tool calls, and LLM thinking in real time.
- **Two LLM backends**
  - **Google AI Studio** (hosted, API key required)
  - **Ollama** (self-hosted, e.g. on a free Kaggle GPU, exposed through a Pinggy tunnel)
- **Markdown reports** rendered directly in the app.
- Telemetry for CrewAI / OpenTelemetry is disabled by default.

---

## 🧠 How It Works

```
┌──────────────┐     ┌─────────────────────────────────────────────┐
│  Flet UI     │────▶│  CrewAI                                     │
│  (finale.py) │     │                                             │
└──────┬───────┘     │  Crop mode  (hierarchical process)          │
       │             │   ├─ Agricultural Research Manager          │
       │             │   ├─ Location Finder ──▶ Get Coordinates    │
       │             │   └─ Climate Data Analyst ──▶ NASA POWER    │
       │             │                                             │
       │             │  Query mode (sequential process)            │
       │             │   └─ Agricultural AI Assistant (both tools) │
       │             └──────────────────┬──────────────────────────┘
       │                                │
       │                      ┌─────────▼─────────┐
       └──────────────────────│  LLM backend      │
         live event feed      │  Google / Ollama  │
                              └───────────────────┘
```

### Tools available to the agents

| Tool | Purpose |
|------|---------|
| `Get Coordinates Tool` | Geocodes a place name to latitude/longitude via Nominatim (`geopy`). |
| `fetch_nasa_power_agricultural_data` | Fetches daily NASA POWER data (community `AG`) for a coordinate and date range, returned as CSV. Missing values (`-999`) become `NaN`; dates are flexibly normalized to `YYYYMMDD`. |

### Parameters fetched from NASA POWER

`T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`, `WS2M`, `T2MDEW`, `GWETTOP`, `GWETROOT`, `GWETPROF`

---

## 📁 Project Structure

```
.
├── finale.py                    # The AgriSense Flet application
├── notebooke2207dbd45.ipynb     # Kaggle notebook: runs Ollama + Pinggy tunnel
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (3.11+ recommended)
- Internet access (NASA POWER, Nominatim, and your chosen LLM provider)
- One of:
  - a **Google AI Studio API key**, or
  - an **Ollama endpoint** (see [Using Ollama on Kaggle](#-using-ollama-on-kaggle-via-pinggy))

### Installation

```bash
# 1. Clone / download the project, then:
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install flet crewai geopy requests pandas
```

> Note: the code uses the newer Flet API (`ft.run`, `ft.Padding`, `ft.Border`, `ft.Alignment`), so make sure you have a recent Flet release (0.80+). If you run into attribute errors, upgrade with `pip install -U flet`.

### Run

```bash
python finale.py
```

The desktop window opens (1400×880).

---

## ⚙️ Configuration

Click the **⚙️ settings icon** in the top-right corner to choose your model backend. Settings are held in memory for the session only (nothing is written to disk).

### Option A — Google AI Studio

1. Create an API key at [Google AI Studio](https://aistudio.google.com/).
2. In **Model Settings**, select **Google AI Studio**.
3. Paste your **Google API Key** and set the **Google Model** name (default: `gemma-4-27b-it`).

### Option B — Ollama (via Pinggy)

1. Start Ollama on Kaggle using the notebook (see below) and copy the public URL it prints.
2. In **Model Settings**, select **Ollama (Pinggy)**.
3. Paste the **Pinggy URL** (e.g. `https://xxxx.run.pinggy-free.link`) and set the **Ollama Model** (default: `gemma4`).

---

## 🖥️ Usage

### Crop Analysis

1. Choose **Crop Analysis**.
2. Enter a **Location** and a **Crop**.
3. Use the slider to select the **Analysis Duration** (1–365 days, counted back from today).
4. Click **Run Analysis**.

The report includes the location and coordinates, a NASA climate summary (averages, totals, trends), climate analysis, a suitability verdict for your time window, and alternative crops if needed.

### Custom Query

1. Choose **Custom Query**.
2. Type your question and click **Run Analysis**.

If your question involves a place, the agent geocodes it and pulls the last 30 days of climate data by default; otherwise it answers directly.

---

## ☁️ Using Ollama on Kaggle via Pinggy

The notebook `notebooke2207dbd45.ipynb` turns a Kaggle session into a free, publicly reachable Ollama server.

**Steps performed by the notebook:**

1. Installs system dependencies and **Ollama**.
2. Installs the `pinggy` Python package.
3. Starts `ollama serve` bound to `0.0.0.0:11434` (CORS open, model kept loaded with `OLLAMA_KEEP_ALIVE=-1`) and waits until it is ready.
4. Pulls the `gemma4` model.
5. Opens a **Pinggy HTTPS tunnel** to port `11434` (rewriting the `Host` header to `localhost:11434`) and verifies it responds.
6. Sends a test chat request to confirm end-to-end operation.

**To use it:**

1. Upload the notebook to Kaggle, enable a **GPU accelerator** and **Internet**.
2. Run all cells in order and copy the printed **Public URL**.
3. Paste that URL into AgriSense under **Settings → Ollama (Pinggy)**.

> ⚠️ Keep the notebook session running while using the app. Free Pinggy URLs are temporary and change each session.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | [Flet](https://flet.dev/) |
| Agent framework | [CrewAI](https://www.crewai.com/) |
| Climate data | [NASA POWER API](https://power.larc.nasa.gov/) |
| Geocoding | [geopy](https://geopy.readthedocs.io/) + OpenStreetMap Nominatim |
| Data handling | pandas, requests |
| LLMs | Google AI Studio, Ollama |
| Tunneling | [Pinggy](https://pinggy.io/) |

---

## 🔧 Troubleshooting

| Problem | Likely cause / fix |
|---------|--------------------|
| *"Google API key is not set."* | Open Settings and enter your key. |
| *"Pinggy / Ollama URL is not set."* | Paste the tunnel URL in Settings. |
| Ollama request fails | The Kaggle session or tunnel expired — rerun the notebook and use the new URL. |
| *"Location not found."* | Use a cleaner place name, e.g. `Lahore, Pakistan`. |
| NASA POWER returns no data | Check coordinates and date range; very recent days may not be available yet. |
| Long runs / timeouts | Large windows (up to 365 days) and small models take longer; LLM timeout is 600 s, NASA request timeout is 300 s. |
| Flet attribute errors | Upgrade Flet: `pip install -U flet`. |

---

## ⚠️ Disclaimer

AgriSense provides AI-generated suggestions based on recent climate data. It is intended as a decision-support aid, **not** a substitute for local agronomic expertise, soil testing, or professional advice. Crop suitability also depends on factors not covered here (soil type, irrigation, market, pests, etc.).

---

## 🙏 Acknowledgements

- [NASA POWER](https://power.larc.nasa.gov/) for agroclimatology data
- [CrewAI](https://www.crewai.com/), [Flet](https://flet.dev/), [Ollama](https://ollama.com/), [Pinggy](https://pinggy.io/)
- OpenStreetMap / Nominatim for geocoding

## 📄 License

Add your preferred license here (e.g. MIT).
