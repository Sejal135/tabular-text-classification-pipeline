# Transaction Classification

Six-class classification over 5,899 anonymized accounting transactions, with an
interactive dashboard and an on-demand agentic prediction explainer.

**Headline result:** cross-validated macro F1 of **0.7402 ± 0.0242**, against a
majority-class baseline of **0.1877**. Accuracy is 0.938 against a baseline of
0.884, which is exactly why accuracy is not the reported metric.

---

## Setup

Requires Python 3.11 or newer (developed on 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `Challenge_Data.xls` in `data/raw/` if it is not already there.

The explainer calls a language model through Groq's free tier. Copy the example
env file and add a key from console.groq.com (email signup, no card):

```bash
cp .env.example .env
```

Without a key the explainer still runs, composing the same measured evidence
into prose locally instead of calling the model. That fallback exists so this
project is testable without credentials, not as a substitute for the agent.

## Run the dashboard

```bash
streamlit run app/Home.py
```

Opens at http://localhost:8501 with two tabs:

- **Analysis and model** — exploration, before/after preprocessing views, split
  strategy, and model evaluation. All charts are computed live from the data.
- **Score new data** — upload a file with columns Col1 to Col7 and get
  predictions. Select any row and press Explain for a per-row explanation.

To try the scoring tab immediately, upload `data/holdout/holdout_features.csv`.
Including a label column also lights up live accuracy and a confusion matrix.

## Reproduce the model

The saved artifacts are already in `artifacts/`. To regenerate everything, run
the notebooks in order:

1. `notebooks/01_eda.ipynb` — exploration, and the findings behind every
   feature decision, including the ones that were tested and rejected
2. `notebooks/02_modeling.ipynb` — split, model comparison, tuning, final fit,
   and the single holdout evaluation

Both are deterministic. Every random seed is fixed, so a rerun reproduces the
numbers quoted here exactly.

---

## Structure

```
data/raw/            Challenge_Data.xls
data/holdout/        the 10% holdout, features and labels kept separate
notebooks/           01_eda.ipynb, 02_modeling.ipynb
src/
  preprocessing.py   cleaning, derived features, the fitted pipeline
  splits.py          group-aware CV splitter
  evaluate.py        cross-validation scoring and the macro-F1 scorer
  two_stage.py       two-stage classifier (built, tested, rejected)
  explain.py         per-row feature attribution and evidence assembly
  agent.py           LLM agent with tool use, plus the offline fallback
app/
  Home.py            entrypoint
  common.py          shared loaders and upload handling
  charts.py          plotly chart builders
  tab_analysis.py    tab 1
  tab_score.py       tab 2, including the explainer panel
artifacts/           pipeline.joblib, metrics.json, figures, result tables
```

---

## The model

A single scikit-learn `Pipeline`, saved whole to `artifacts/pipeline.joblib`:

```
RawPrep  ->  ColumnTransformer  ->  RandomForestClassifier
```

`RawPrep` runs cleaning and feature derivation inside the pipeline, so the
saved object accepts raw Col1 to Col7 and returns a prediction with no manual
preprocessing. That is what lets the dashboard score an arbitrary uploaded CSV,
and what makes the notebook and the app provably agree.

**Classifier.** Random forest, 433 trees, `class_weight="balanced"`,
`max_features="log2"`, `min_samples_leaf=1`, unbounded depth.

**Features, 876 columns from 7 input columns:**

| Input | Treatment | Reason |
|---|---|---|
| Col1, Col4, Col6 | TF-IDF over whitespace tokens, `min_df` 5 / 5 / 3 | Values share tokens. One token, Word575, appears across three separate Col4 values totalling 114 rows, all Category_2. Encoding whole values forces the model to learn it three times from 56, 50, and 8 rows. |
| Col6 | first token, one-hot, alongside its TF-IDF | 196 values collapse to 24 first tokens, and two of them hold most of Category_4 and Category_6. TF-IDF discards position; the hierarchy lives in position. |
| Col2 | raw value dropped; kept as is_numeric, length, and a character shape mask | 88% of its 4,818 values appear exactly once. As a category it memorizes rows. Digit-only codes are nine times more likely to be Category_2, so the structure is kept and the identity discarded. |
| Col3 | signed log, is_round_amount, is_negative, median imputer, scaler | Skew 8.39 with 69 negative values, so a plain log transform fails. |
| Col5 | dropped | 17 distinct dates in total, the lowest mutual information of any column, and removing it moved the score by 0.005 against seed noise of 0.05. |
| Col7 | one-hot | Retained, though Cramer's V of 0.961 against Col6 shows it is largely a coarser version of it. |

**Selection.** Balanced class weights were the single largest gain in the whole
project, moving macro F1 from 0.594 to 0.755. Hyperparameter search over 40
candidates added 0.013 on top of that. Two-stage models, per-class decision
thresholds, a Col4 missingness indicator, and deduplication were each built,
measured, and rejected; the reasons are in `01_eda.ipynb` and
`02_modeling.ipynb`.

**Validation.** Group-aware stratified splitting throughout. Rows with identical
features share a group key and always move together, so no duplicate straddles
the train/test boundary. Three cross-validation folds, repeated across three
seeds, because fold-to-fold variance on the rare classes exceeds the difference
between most modeling choices.

## The explainer

Selecting a row and pressing Explain runs a two-layer process.

**Evidence first.** `explain.py` measures what actually drove the prediction by
leave-one-feature-out ablation: each active feature is zeroed in turn and the
drop in predicted probability recorded. It also pulls how each token in the row
behaved during training, and the model's known weaknesses for the predicted
class. SHAP is supported but gated behind an additivity check, which it fails
on this model.

**Prose second.** `agent.py` sends that evidence to a language model with three
tools it can call: token lookup, class reliability profile, and model overview.
The model decides which lookups it needs, then writes the explanation. It is
instructed never to state a number that did not come from the evidence or a
tool result, and never to guess what an anonymized token means.

The dashboard shows the written explanation, the attribution tables behind it,
and the agent's tool calls, so every sentence can be checked against the data
beneath it.

## Known limitations

- **Category_5 has two rows in the entire dataset.** It is never validated, so
  no reliability figure for it exists. Both rows are kept in every training
  fold.
- **Category_4 and Category_6 have 14 and 11 training rows.** Their F1 scores
  of 0.500 and 0.545 carry wide fold-to-fold variance and are reported as
  ranges rather than point estimates.
- **The holdout scores higher than cross-validation**, 0.9003 against 0.7402.
  This is an artifact, not an improvement: three of the five classes in the
  holdout have one or two rows each. The classes with real sample sizes agree
  closely with cross-validation, and that agreement is the actual evidence of
  generalization.
- **Ablation under-counts correlated features.** Removing one of two features
  carrying the same information leaves the other in place, so each looks weaker
  alone than they are together.
- **The text is anonymized.** Word575 is an opaque token, so no semantic or
  pretrained-embedding approach applies and no feature can be interpreted in
  business terms.