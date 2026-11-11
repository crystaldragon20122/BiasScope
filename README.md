# BiasScope

**Automated, rubric-based bias testing for a career-advice AI.**

BiasScope asks one narrow question: *if two fictional students are identical on paper except for the gender
they state, does a career-advice AI treat them differently?* It generates matched test cases, runs them
against the AI being tested, has a separate judge model compare the answers against an explicit rubric, and
reports **potential** bias with quoted, verified evidence.

> The generate → test → evaluate → report workflow is inspired by [Giskard](https://www.giskard.ai/)'s LLM scan.
> BiasScope is an independent implementation built for a TKS student project. It does not wrap Giskard and
> does not reproduce its internals.

It never certifies a model as fair. A clean scan means "nothing found under this rubric, with this judge, on
these profiles" – nothing more.

---

## Quick start (offline demo – no API key, no internet)

You need **Python 3.10–3.13**. Check with `python --version` (Windows) or `python3 --version` (macOS).

### macOS (Terminal)

```bash
cd biasscope
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Windows (PowerShell)

```powershell
cd biasscope
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
Command Prompt: `.venv\Scripts\activate.bat`.

The browser opens at <http://localhost:8501> on the **Home** page. Use the menu along the top:

| Page | What it's for |
|---|---|
| **Home** | Title page: what BiasScope is and where to start |
| **Try the demo** | One-click scan on a *pretend* advisor (biased or fair). Free, offline, no key |
| **Live proof** | A real AI judge grades 8 hand-written answer pairs (4 biased, 4 fair), each twice with A/B swapped – 16 small calls, about 1 cent. A free word-matching "practice run" needs no key |
| **Run a scan** | The full test of a real AI model (or the pretend one), step by step, with provider presets for OpenAI, Claude and Ollama |
| **Results** | Counts out of totals, a click-to-zoom sunburst (gender → bias type → profile), tables, downloads, run-to-run comparison |
| **Evidence** | Two answers side by side with the flagged sentence highlighted; Previous / Next to step through |
| **Check the judge** | Tests the judge on 15 hand-written examples: recall, precision, unsure answers, and whether swapping A/B changes its mind |
| **How it works** | The method and its limitations in plain words |

Demo results are labelled **SIMULATED**: the advisor is scripted and the judge returns pre-set labels. They show
how the pipeline works; they do not evaluate any real AI.

### Headless (no browser)

```bash
python -m biasscope demo --behavior stereotyped   # writes suite/results JSON, CSV and a Markdown report
python -m biasscope demo --behavior neutral
python -m biasscope validate                      # keyword-baseline auditor validation
```

(On Windows use `py -m biasscope ...` if `python` isn't on your PATH.)

### Run the tests

```bash
python -m pytest        # macOS: python3 -m pytest if the venv isn't active
```

---

## Live mode (real models)

Live mode talks to any server that implements the OpenAI-style `POST {base_url}/chat/completions` endpoint.

**Cost and hardware warning.** Hosted APIs (OpenAI, OpenRouter, Together, Groq, …) usually **charge per
token**. A default scan is 90 target calls + up to 90 judge calls, and retries can add more. Local servers
(Ollama, LM Studio, vLLM) cost nothing per call but need a computer able to run the model – roughly 8 GB+
of RAM for a small 7–8B model, more for larger ones. A chat **subscription** (Claude Pro, ChatGPT Plus, …)
does **not** by itself include API access; API keys are billed separately.

### Option A – a hosted API

1. Create an API key with your provider and add billing if required.
2. Either copy `.env.example` to `.env` and fill in the values, or type them into the app (password fields
   are kept only for the browser session).
3. Open **Run a scan**, choose *A real AI model*, then:
   - **The AI being tested**: pick a provider (OpenAI, Claude, Ollama or Other). The preset fills in the base URL,
     a sensible model and the right settings (e.g. temperature off for Claude). Paste your key.
   - **The judge**: ideally a *different*, capable model, kept the same across scans you want to compare.
   - **Advanced settings** (optional): newer OpenAI reasoning models need `max_completion_tokens`; untick
     *Send temperature* for models that reject it.
4. Read the planned number of requests, tick the cost box, and click **Run scan**. Try 2 profiles × 1 repeat first.

### Option B – a local model with Ollama (free per call)

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b
ollama serve            # often already running in the background
```

Base URL `http://localhost:11434/v1`, model `llama3.1:8b`, API key blank. Small local models are weak judges –
check them on the **Check the judge** page before trusting their verdicts.

### Generating new profiles

On **Run a scan**, choose *Write new profiles with an AI* as the profile source. The generator only
writes demographic-neutral profiles; every one is schema-checked and **rejected** if it contains names,
pronouns or gendered words. Gender is added afterwards by code. Rejections are listed, never silently fixed.

### Replaying a suite against another model

Tests (suites) are saved to `workspace/suites/` and can be downloaded. Choose *A saved test* or *Upload a test
file* to run the identical cases against a different AI. On **Results → Downloads & compare**,
BiasScope refuses to compare runs from different suites and lists every configuration difference.

### Putting the website online (Streamlit Community Cloud)

1. Upload the project folder's contents to a GitHub repository – **not** `.venv`, `workspace` or `.env`.
2. At share.streamlit.io: *Create app → Deploy a public app from GitHub*, pick the repo, main file `app.py`,
   Python 3.12.
3. **Never put API keys in Streamlit's Secrets** – visitors could then spend your credit. Visitors type their own
   keys, which stay in their browser session.
4. Free apps go to sleep when unused; the first visit after a while can take a minute to wake up. Open the link
   yourself shortly before sharing or presenting.
5. Scans on the hosted site are visible to all its visitors and are wiped when the app restarts.

### Credentials and safety

- Keys come from environment variables (`.env`) or session-only password fields.
- Keys are excluded from every data structure that is saved, and every export is scrubbed again as a safety
  net. If a server echoes your key back, it is redacted before saving.
- Model output is always HTML-escaped before display and is never executed.

---

## Project layout

```
app.py                        website entry point: page setup + top navigation
ui_common.py                  shared look, logo, provider presets and run helpers
views/                        one file per page (home, demo, live_proof, scan, results, evidence,
                              judge_check, about)
assets/                       logo (SVG/PNG) and favicon
biasscope/
  config.py                   constants, paths, extensibility registry (domains/attributes)
  schemas.py                  Pydantic models: profiles, suites, provider configs, records, judgments
  scenarios.py                neutrality check, deterministic gender insertion, suite save/load/verify
  generation.py               live profile generation with validation and rejection reporting
  providers.py                OpenAI-compatible HTTP client, retries/backoff, simulated advisor & judge
  execution.py                target runs: fresh conversations, seeded order, incremental saving
  judging.py                  pairwise A/B judging, prompt-injection defences, evidence verification
  baseline.py                 keyword baseline auditor (offline, non-semantic)
  analysis.py                 descriptive counts with denominators, run-to-run compatibility
  reporting.py                JSON / CSV / Markdown exports, secret scrubbing, safe highlighting
  validation.py               fixture validation, precision/recall, order sensitivity, label import
  storage.py                  JSON/JSONL run storage
  pipeline.py                 orchestration (create, run, resume)
  live_proof.py               8-pair live proof: grading, order check, summary
data/
  curated_profiles.json       12 neutral fictional profiles
  mock_advisor_content.json   text building blocks for the SIMULATED advisor
  validation_fixtures.json    15 fixtures with developer-authored reference labels
  prompt_tuning_examples.json kept separate from validation fixtures
  live_proof_pairs.json       4 biased + 4 fair answer pairs for the Live proof page
docs/
  EXPERIMENT.md               plain-language explanation and limitations
  DEMO_SCRIPT.md              two-minute TKS demonstration outline
tests/                        pytest suite (73 tests)
```

Adding another attribute or domain later means adding an entry in `config.ATTRIBUTES` / `config.DOMAINS`, a
prompt template, and widening the `Variant` type – the pipeline itself is attribute-agnostic.

See **docs/EXPERIMENT.md** for how the experiment works and its limitations.
