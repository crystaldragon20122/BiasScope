import streamlit as st

import ui_common as ui
from biasscope.schemas import CRITERIA, CRITERION_LABELS

ui.page_header("How it works", "The whole experiment in plain words.", kicker="Method")

st.markdown("""
### The question
If two students are identical on paper except for the gender they state, does an AI career advisor treat them
differently? This is called a **matched-pair test**: hold everything else still, change one thing, and see what moves.

### 1 · Make the twins
Each fictional student has an education level, strengths, grades, interests, experience and goals – but **no name,
no pronouns and no gendered words**. A checker throws out any profile that sneaks one in. Then the code (not an AI)
makes three copies that differ in exactly one line: `Gender: woman`, `Gender: man`, or `Gender: nonbinary`.

### 2 · Ask the AI
Each copy is sent to the AI being tested several times, each time in a **brand-new chat**, so earlier answers can't
leak into later ones. The order is shuffled with a fixed seed so the scan can be repeated exactly.

### 3 · Judge in pairs
A second AI – the **judge** – reads two answers at a time, labelled only A and B (a coin flip decides which is which),
and never learns which AI wrote them. It checks four kinds of bias:
""")
for cid, desc in CRITERIA.items():
    st.markdown(f"- **{CRITERION_LABELS[cid]}** – {desc.split(': ', 1)[1]}")
st.markdown("""
Different wording, a different order, or one different career idea are **not** bias on their own.

For every flag, the judge must **quote the exact sentence**. BiasScope checks the quote really appears in that answer.
If it doesn't, the flag is downgraded to "uncertain" – a made-up quote never counts. The judge is also told the answers
are just data and to ignore any instructions hidden inside them.

### 4 · Show the proof
Every number is shown as **count out of total**. Refusals, broken answers, errors and unsure verdicts are counted
separately and never counted as "fair". There's deliberately **no single fairness score**, because squashing different
kinds of harm into one number hides more than it shows.

### What the results can't tell you
- A flag is **potential** bias – a lead to read and check, not proof. No flags isn't proof of fairness either.
- The judge can be wrong. That's why there's a **Live proof** page and a **Check the judge** page.
- It's a small, fictional sample in one area (career advice, in English). Repeats of one profile aren't independent,
  so BiasScope reports counts, not statistical significance.
- It only tests a **stated** gender. Bias can also show up through names, writing style or patterns across many
  answers, which this version doesn't measure.
- Results depend on the model version, its instructions and the date – all saved with every scan.

### Credits
The generate → test → evaluate → report workflow is inspired by **Giskard**'s open-source AI scan. BiasScope is an
independent implementation and doesn't reproduce Giskard's internals.
""")
ui.footer()
