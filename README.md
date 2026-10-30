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

The browser opens at <http://localhost:8501>. Then:

1. **Setup and Run** → leave *Offline demo* selected → keep *Advisor with deliberate, obvious stereotypes* →
   click **▶ Run Scan** (takes about a second).
2. **Results** → counts, rates with denominators, charts, downloads.
3. **Evidence Explorer** → pick a finding and read the two matched responses side by side with the evidence
   highlighted.
4. **Auditor Validation** → click **Run validation** to check the (offline) keyword baseline on the fixtures.

Everything in offline mode is labelled **SIMULATED**: the advisor is scripted and the judge returns
predetermined demonstration labels. It shows how the pipeline works; it does not evaluate any real AI.

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
3. In **Setup and Run** choose *Live*, then set:
   - **Target**: the model you want to test, e.g. base URL `https://api.openai.com/v1` and a model name.
   - **Judge**: ideally a *different*, capable model. Temperature 0 is recommended.
   - **Token-limit parameter**: newer OpenAI models expect `max_completion_tokens`; most other servers accept
     `max_tokens`. If a model rejects `temperature`, untick *Send temperature*.
4. Read the planned call count, tick the cost acknowledgement, and click **▶ Run Scan**.

### Option B – a local model with Ollama (free per call)

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1:8b
ollama serve            # often already running in the background
```

Base URL `http://localhost:11434/v1`, model `llama3.1:8b`, API key blank. Small local models are weak judges –
check them on the **Auditor Validation** page before trusting their verdicts.

### Generating new profiles

With a generator model configured, choose *Generate new profiles* as the suite source. The generator only
writes demographic-neutral profiles; every one is schema-checked and **rejected** if it contains names,
pronouns or gendered words. Gender is added afterwards by code. Rejections are listed, never silently fixed.

### Replaying a suite against another model

Suites are saved to `workspace/suites/` and can be downloaded. Choose *Load a saved suite* or *Upload a suite
JSON* to run the identical cases against a different target. On **Results → Compare with another run**,
BiasScope refuses to compare runs from different suites and lists every configuration difference.

### Credentials and safety

- Keys come from environment variables (`.env`) or session-only password fields.
- Keys are excluded from every data structure that is saved, and every export is scrubbed again as a safety
  net. If a server echoes your key back, it is redacted before saving.
- Model output is always HTML-escaped before display and is never executed.

---

## Project layout

```
app.py                        Streamlit UI (Setup and Run, Results, Evidence Explorer, Auditor Validation)
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
data/
  curated_profiles.json       12 neutral fictional profiles
  mock_advisor_content.json   text building blocks for the SIMULATED advisor
  validation_fixtures.json    15 fixtures with developer-authored reference labels
  prompt_tuning_examples.json kept separate from validation fixtures
docs/
  EXPERIMENT.md               plain-language explanation and limitations
  DEMO_SCRIPT.md              two-minute TKS demonstration outline
tests/                        pytest suite (65 tests)
```

Adding another attribute or domain later means adding an entry in `config.ATTRIBUTES` / `config.DOMAINS`, a
prompt template, and widening the `Variant` type – the pipeline itself is attribute-agnostic.

See **docs/EXPERIMENT.md** for how the experiment works and its limitations.
