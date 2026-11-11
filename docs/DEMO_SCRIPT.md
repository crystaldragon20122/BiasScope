# Demo outline (2–3 minutes)

**Before you start:** open the site a minute early (free hosting can take a moment to wake up). Have a demo scan
already run with the *biased* advisor, and one with the *fair* advisor.

| Time | Show | Say (roughly) |
|---|---|---|
| 0:00–0:20 | **Home** | "AI career advisors talk to millions of students. BiasScope asks: if two students are identical except for the gender they state, does the AI treat them the same? It's inspired by Giskard's scan workflow, but I built the pipeline myself." |
| 0:20–0:45 | **Try the demo** → *Biased advisor* → **Run the demo scan** | "Profiles have no names or pronouns. Code adds one line – woman, man or nonbinary – so the questions differ in exactly one line. This demo uses a pretend advisor, so it's free." |
| 0:45–1:15 | **Results** → click a gender in the sunburst, then a bias type | "Every number is count out of total. Errors and unsure verdicts are never counted as fair. Clicking zooms in: here's whose answers were flagged, for which kind of bias, on which profiles." |
| 1:15–1:40 | **Evidence** | "The highlighted sentence is quoted exactly from the answer. If the judge invents a quote, BiasScope catches it and downgrades the flag." |
| 1:40–2:20 | **Live proof** → run with a real judge (or the practice run) | "Does the checker actually work? A real AI judge gets 4 biased and 4 fair pairs, each twice with the order swapped. It caught X of 8 and left Y of 8 fair checks alone. The free word-matching checker only catches the obvious one." |
| 2:20–2:45 | **Run a scan** (walk through the steps) | "To test a real model you pick a provider, paste a key, pick a judge and run. Here are my results for [models]." |
| 2:45–3:00 | **How it works** → limitations | "Findings are *potential* bias – leads for a human to read, not a fairness certificate." |

**If asked "did you test a real model?"** – only say yes if you actually ran a live scan, and show that scan on
Results (it's labelled *Live run*, with the model name).
