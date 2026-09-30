# Third-party code and data, not redistributed here

The experiments depend on the following. We link them rather than bundle them,
so that their own licences and versions govern.

| Component | Where | Used for |
|---|---|---|
| CIC-IDS2017 flow dataset | Canadian Institute for Cybersecurity, University of New Brunswick | the traffic the detector is trained and evaluated on |
| scikit-learn | scikit-learn.org | the random-forest detector and the isotonic calibrator |
| Groq API | groq.com | the LLM attacker arm |
| `openai/gpt-oss-20b` | served by Groq | the attacker model in the corrected run |
| `llama-3.1-8b-instant` | served by Groq | the attacker model in the superseded v6 round |

Record the version or commit of each before re-running; the scripts do not pin them.
