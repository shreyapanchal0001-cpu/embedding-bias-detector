# Embedding Bias Detector

**Measure demographic bias in the embedding spaces of generative AI models.**

A lightweight Python tool that quantifies how much an AI model's internal text representations associate concepts unevenly across demographic groups — and reports the result as a colour-coded bias percentage.

---

## Why This Exists

Modern AI systems don't "see" words — they see vectors. When a model encodes `nurse` and `engineer` into numerical space, it inherits the statistical patterns of its training data, including societal stereotypes.

Most fairness toolkits audit **model outputs** (predictions, classifications, decisions). This project audits the **representation layer** — the embedding space itself — where bias originates *before* any output is generated.

The goal is a tool that drops into a data scientist's existing workflow: load a model, point it at a set of concept groups, and get a readable bias report in seconds.

---

## How It Works

The method is a centroid-difference association test, in the tradition of WEAT (Word Embedding Association Test, Caliskan et al. 2017).

**Step 1 — Embed.** Every term is converted into a dense vector by a sentence-transformer model.

```
"engineer"  →  [0.567, 0.234, 0.789, ..., 0.123]   (384 dimensions)
```

**Step 2 — Build group centroids.** Terms belonging to each demographic group are averaged into a single representative vector.

```
centroid_male   = mean(embed("man"), embed("he"), embed("husband"), ...)
centroid_female = mean(embed("woman"), embed("she"), embed("wife"), ...)
```

**Step 3 — Measure association.** For each target concept, compute the difference in cosine similarity to each centroid.

```
association_score = cos(target, centroid_A) − cos(target, centroid_B)
```

A positive score leans toward group A, a negative score toward group B, and zero indicates neutrality.

**Step 4 — Aggregate to a percentage.** The mean absolute deviation from neutrality is scaled onto a 0–100% band.

```
bias % = mean(|scores|) / 0.3 × 100
```

**Interpretation bands:**

| Band | Status |
|------|--------|
| 0–20% | 🟢 Low bias |
| 20–40% | 🟡 Moderate bias |
| 40%+ | 🔴 High bias |

---

## Quickstart

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/shreyapanchal0001-cpu/embedding-bias-detector.git
cd embedding-bias-detector

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
cd examples
python run_first_test.py
```

> **Note:** The first run downloads the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face and caches it locally. Subsequent runs are fast.

---

## Example Output

```
=======================================================
EMBEDDING BIAS REPORT — Gender x Occupations
=======================================================
Occupation          Score   Leans Toward
-------------------------------------------------------
doctor           -0.0084   Female-associated
programmer       -0.0175   Female-associated
CEO              -0.0204   Female-associated
engineer         -0.0384   Female-associated
scientist        -0.0393   Female-associated
-------------------------------------------------------
Overall Disparity: 23.3%
Status: [WARNING]  🟡 Moderate bias
```

---

## Project Structure

```
embedding-bias-detector/
├── core/
│   ├── __init__.py
│   ├── detector.py       # Embedding engine — loads model, produces vectors
│   └── metrics.py        # Bias math — association scores, disparity percentage
├── config/
│   └── groups.yaml       # Demographic group definitions and target concepts
├── examples/
│   └── run_first_test.py # End-to-end demonstration script
├── tests/                # Test suite (in progress)
├── visualization/
│   ├── report.py         # Dashboard renderer
│   └── bias_report.png   # Generated output
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Configuration

Groups and target concepts are defined in `config/groups.yaml` — no code changes required to test a new hypothesis.

```yaml
gender:
  name: "Gender"
  group_a:
    label: "Male-associated"
    terms: ["man", "he", "his", "husband", "father", "brother", "son", "male"]
  group_b:
    label: "Female-associated"
    terms: ["woman", "she", "her", "wife", "mother", "sister", "daughter", "female"]

attributes:
  name: "Occupations"
  terms: ["engineer", "nurse", "programmer", "teacher", "doctor",
          "receptionist", "CEO", "secretary", "scientist", "babysitter"]
```

**Design guidance for group definitions:**

- Keep group sizes balanced — unequal counts skew the centroids
- Use semantically unambiguous terms (`mother` over `actor`)
- Aim for 6–10 terms per group
- Document your choices, since group selection is a judgement call, not a neutral act

---

## Choosing an Embedding Model

The default model is `all-MiniLM-L6-v2` (Microsoft) — small, fast, and free. Swap it by changing one line in `core/detector.py`:

```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
```

| Model | Size | Notes |
|-------|------|-------|
| `all-MiniLM-L6-v2` | ~80 MB | Default. Fast, good general quality |
| `paraphrase-multilingual-MiniLM-L12-v2` | ~450 MB | 50+ languages |
| `multi-qa-MiniLM-L6-cos-v1` | ~80 MB | Tuned for retrieval |

Comparing results across models is itself informative — different training corpora encode different biases.

---

## Limitations & Honest Caveats

This tool measures disparity. It does **not** determine fairness. Some important limits:

**Sign interpretation is unreliable in isolation.** In the sample output above, every occupation scored negative. This reflects where the group centroids sit in the space, not necessarily a uniform stereotype. The *ranking* of concepts relative to each other carries more signal than the absolute sign. More robust variants (permutation testing, effect-size normalisation) are on the roadmap.

**Group definitions are a human choice.** Who counts as a group, and which terms represent them, shapes the result. This is a known and unavoidable property of fairness measurement — transparency about the choice matters more than pretending it's objective.

**No statistical significance testing yet.** Current output is descriptive. Confidence intervals and p-values are planned.

**Embedding bias ≠ output bias.** A model can have biased internal representations that are mitigated downstream, or vice versa. This tool addresses one layer of a larger system.

**Not a compliance artefact.** Do not use this as sole evidence for regulatory or legal determinations.

---

## Roadmap

- [x] Text embedding pipeline
- [x] Centroid-difference association metric
- [x] Colour-coded bias reporting
- [ ] Visualisation module — heatmaps and ranked bar charts
- [ ] Permutation testing for statistical significance
- [ ] Expanded predefined groups (race, age, disability, religion)
- [ ] Custom group definitions via config
- [ ] Dataset input — audit a CSV with a labelled sensitive column
- [ ] Image and multimodal embeddings (CLIP)
- [ ] CLI interface
- [ ] Test suite with published benchmark validation

---

## Contributing

Issues, critiques, and methodological corrections are genuinely welcome — particularly from anyone working in fairness research. If you spot a flaw in the metric, please open an issue rather than a silent fork.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- Caliskan, Bryson & Narayanan (2017). *Semantics derived automatically from language corpora contain human-like biases.* Science.
- May, Wang, Bordia, Bowman & Rudinger (2019). *On Measuring Social Biases in Sentence Encoders.* NAACL.
- Mehrabi, Morstatter, Saxena, Lerman & Galstyan (2021). *A Survey on Bias and Fairness in Machine Learning.* ACM Computing Surveys.
