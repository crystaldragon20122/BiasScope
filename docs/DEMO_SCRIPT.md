# Two-minute TKS demo outline

**Before you start:** run `streamlit run app.py`, keep the browser open on *Setup and Run*, offline mode,
"Advisor with deliberate, obvious stereotypes". Optionally do one run beforehand with the *neutral* advisor
so you can compare.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:20 | Title / Setup page | "AI career advisors talk to millions of students. I built BiasScope to test one question: if two students are identical except for the gender they state, does the AI treat them differently? It's inspired by Giskard's scan workflow, but I built the pipeline myself." |
| 0:20–0:45 | Expand *Preview profiles and a matched set* | "Profiles have no names or pronouns. Code adds one line – woman, man or nonbinary – so the prompts differ in exactly one line. That isolates gender." |
| 0:45–1:00 | Point at *Planned calls*, click **Run Scan** | "Each prompt goes to the AI three times in fresh conversations; a judge compares every pair of answers against a four-part rubric. This demo is **simulated**, so it's free and runs offline." |
| 1:00–1:25 | **Results** | "Every number has a denominator. Errors, refusals and *uncertain* verdicts are counted separately – never as passes. There's deliberately no single fairness score." |
| 1:25–1:45 | **Evidence Explorer**, open a stereotyping finding | "Here's the evidence: the highlighted sentence is quoted exactly from the answer. If the judge invents a quote, BiasScope catches it – this one was downgraded to uncertain." (filter Verdict = uncertain to show it) |
| 1:45–2:00 | **Auditor Validation**, click Run validation | "The judge can be wrong, so I test it on fixtures, including answers that try to trick it, and I check whether swapping A and B changes its mind. Findings are *potential* bias – leads for a human to read, not a fairness certificate." |

**If asked "did you test a real model?"** – only say yes if you actually ran live mode, and show that run's
Results page (it is labelled LIVE, with the model name and settings).
