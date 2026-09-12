"""LLM agent that explains one prediction.

The agent is given measured evidence, never the raw model. It can call tools
to look up facts it needs, then writes the explanation. Numbers come from the
tools; the model supplies only the prose.

Two ways to run:
  - groq    : needs GROQ_API_KEY, free tier, no credit card, doesn't train on my data
  - offline : no key, deterministic template built from the same evidence

Groq was chosen over the paid alternatives for two reasons: the free tier has
no card requirement, and it does not train on prompts, which matters because
the evidence sent to the model contains client data even though the tokens
are anonymized.

Groq speaks the OpenAI wire format, so the same code path also drives a local
Ollama server by swapping base_url. That is the zero-network option.

The offline path is load-bearing, not decoration. Anyone running this project
without an API key must still be able to press Explain and get something
useful, and any provider failure falls back rather than raising.
"""

import json
import os

from src.explain import CLASS_F1, CLASS_ROWS, load_metrics, load_token_stats

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env"))
except ImportError:
    pass                      # dotenv optional; env vars still work

MAX_STEPS = 4

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env": "GROQ_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "env": None,                       # local server, no key
    },
}

SYSTEM = """You explain single predictions from a transaction classification model to a finance reviewer.

Hard rules, in order of importance:
1. Cite features in the order given to you. The first feature listed under "Features supporting the prediction" is the strongest; say so.
2. For each feature you name, state its measured effect using the exact number supplied, in the form "removing it drops the probability to X%". A feature named without its number is useless to the reader.
3. Call lookup_token for every token you intend to name, before you name it. Then quote the training row count and the lift from the tool result.
4. Never state a number that did not come from the evidence or a tool result. Do not estimate, re-round, or infer one.
5. The text columns are anonymized. Word575 is an opaque identifier, not a word. Never guess what a token means.
6. Do not characterize sample sizes in your own words. Give the count and let the reader judge. Call class_profile if you want to comment on reliability.
7. If class_profile returns a known_failure_mode that matches this prediction, say so. A Category_2 prediction with Category_1 as runner-up is the model's most common mistake and the reader needs to know that.
8. Recommend human review only when the evidence supports it: a narrow margin, a rare class, or a known failure mode. Strong, well-supported drivers are a reason not to flag it. Do not hedge by default.
9. Write probabilities as percentages, not decimals.

Write 140 to 200 words of plain prose, no headings, no bullets, no bold. Cover: what was predicted and how strongly, the two or three strongest drivers each with its measured effect, what pushed the other way, and whether a human should check this row. Address the reader directly."""

TOOLS = [
    {
        "name": "lookup_token",
        "description": "Training-set behaviour of one token in one column: how many rows contained it, the class breakdown, and the lift over that class's base rate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "enum": ["Col1", "Col4", "Col6"]},
                "token": {"type": "string", "description": "e.g. Word575"},
            },
            "required": ["column", "token"],
        },
    },
    {
        "name": "class_profile",
        "description": "How reliable the model is for a given class: training rows, cross-validated F1, and known failure modes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string", "description": "e.g. Category_2"},
            },
            "required": ["class_name"],
        },
    },
    {
        "name": "model_overview",
        "description": "Overall model performance and the metric it was selected on.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------------------------------------------------------------- tools

def _tool_lookup_token(column, token, stats):
    """Report how one token behaved in training: row count, class breakdown,
    and lift over the dominant class's base rate. Says plainly when the token
    was never seen, so the agent cannot invent a history for it."""
    if stats is None:
        return {"error": "token statistics unavailable"}
    counts = stats["tokens"].get(column, {}).get(token)
    if not counts:
        return {"column": column, "token": token, "found": False,
                "note": "This token was not seen in the modeling set. The model "
                        "has no evidence about it."}
    total = stats["n_rows"]
    n = sum(counts.values())
    cls, k = max(counts.items(), key=lambda kv: kv[1])
    prior = stats["class_counts"].get(cls, 0) / total
    return {
        "column": column, "token": token, "found": True,
        "rows_in_training": n,
        "class_breakdown": counts,
        "dominant_class": cls,
        "share_of_rows_with_this_token": round(k / n, 3),
        "base_rate_of_that_class": round(prior, 4),
        "lift": round((k / n) / prior, 1) if prior else None,
    }


def _tool_class_profile(class_name, metrics):
    """Report how trustworthy the model is for one class: training rows,
    cross-validated F1, and any known failure mode, so the agent can flag a
    weak prediction instead of presenting every class as equally reliable."""
    n = CLASS_ROWS.get(class_name)
    out = {"class": class_name, "training_rows": n,
           "cross_validated_f1": CLASS_F1.get(class_name)}
    if n is not None and n < 30:
        out["warning"] = ("Very few training examples. Predictions of this class "
                          "are unstable and should be reviewed by a person.")
    if class_name == "Category_5":
        out["warning"] = ("Two rows in the entire dataset, never validated. This "
                          "prediction has no measured reliability.")
    if class_name == "Category_2":
        out["known_failure_mode"] = ("The model's largest error block is "
                                     "Category_1 rows predicted as Category_2, "
                                     "about 5% of all Category_1 rows.")
    if class_name == "Category_3":
        out["known_failure_mode"] = ("Perfect recall but precision 0.808: it "
                                     "absorbs misclassified Category_4 and "
                                     "Category_6 rows.")
    return out


def _tool_model_overview(metrics):
    """Report the model's overall scores and the metric it was selected on,
    for when the agent needs to place one prediction in context."""
    cv = metrics.get("cv", {})
    base = metrics.get("baselines", {})
    return {
        "model": "random forest, balanced class weights, TF-IDF token features",
        "cv_macro_f1_mean": cv.get("macro_f1_mean"),
        "cv_macro_f1_std": cv.get("macro_f1_std"),
        "majority_class_baseline_accuracy": base.get("dummy_accuracy_seed42",
                                                     base.get("dummy_accuracy")),
        "note": ("Selected on macro F1 rather than accuracy, because predicting "
                 "the majority class for every row already scores about 88% "
                 "accuracy."),
    }


def run_tool(name, args, stats, metrics):
    """Dispatch a tool call by name and return its result as plain data.
    An unknown name returns an error dict rather than raising, so a
    hallucinated tool call cannot take the whole explanation down."""
    if name == "lookup_token":
        return _tool_lookup_token(args.get("column"), args.get("token"), stats)
    if name == "class_profile":
        return _tool_class_profile(args.get("class_name"), metrics)
    if name == "model_overview":
        return _tool_model_overview(metrics)
    return {"error": f"unknown tool {name}"}


# ---------------------------------------------------------------- prompt

def evidence_prompt(ev):
    """Format the evidence dict as the user message for the model: prediction
    and runner-up, contributions for and against, token behaviour, class
    caveats, and the raw row. Ordering matters, since the system prompt tells
    the model the first supporting feature listed is the strongest."""
    lines = [
        f"Predicted class: {ev['prediction']} with {ev['agreement']:.1%} of trees agreeing.",
        f"Runner-up: {ev['runner_up']} at {ev['runner_up_probability']:.1%}.",
        "",
        f"Attribution method: {ev['attribution_method']}.",
    ]
    if ev["attribution_method"] == "ablation":
        lines.append("Each contribution below is the drop in predicted "
                     "probability when that single feature is removed from this "
                     "row. Correlated features under-count, because removing one "
                     "leaves the other in place.")
    lines += ["", "Features supporting the prediction:"]
    for f in ev["pushing_toward"]:
        tail = (f", probability falls to {f['probability without it']}"
                if "probability without it" in f else "")
        lines.append(f"  {f['readable']}: {f['contribution']:+.4f}{tail}")

    if ev["pushing_against"]:
        lines += ["", "Features working against the prediction:"]
        for f in ev["pushing_against"]:
            lines.append(f"  {f['readable']}: {f['contribution']:+.4f}")

    if ev["token_evidence"]:
        lines += ["", "How these tokens behaved in training:"]
        for t in ev["token_evidence"]:
            lines.append(
                f"  {t['column']} {t['token']}: {t['rows in training']} rows, "
                f"{t['share']:.0%} {t['dominant class']}, "
                f"{t['lift vs base rate']}x base rate")

    lines += ["", "Known limitations for this class:"]
    lines += [f"  {c}" for c in ev["caveats"]]
    lines += ["", "The row itself:"]
    for k, v in ev["row"].items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Explain this prediction."]
    return "\n".join(lines)


# ---------------------------------------------------------------- offline

def template_explanation(ev):
    """Deterministic fallback. Same evidence, no model, no API key."""
    parts = [
        f"The model predicts {ev['prediction']}, with {ev['agreement']:.1%} of "
        f"the trees agreeing. Its second choice was {ev['runner_up']} at "
        f"{ev['runner_up_probability']:.1%}."
    ]

    top = ev["pushing_toward"][:3]
    if top:
        bits = []
        for f in top:
            if "probability without it" in f:
                bits.append(f"{f['readable']} (removing it drops the probability "
                            f"to {f['probability without it']:.1%})")
            else:
                bits.append(f"{f['readable']} ({f['contribution']:+.3f})")
        parts.append("The strongest support came from " + "; ".join(bits) + ".")

    strong = [t for t in ev["token_evidence"] if t["lift vs base rate"] >= 2]
    if strong:
        t = strong[0]
        parts.append(
            f"In training, {t['column']} containing {t['token']} appeared in "
            f"{t['rows in training']} rows and was {t['dominant class']} "
            f"{t['share']:.0%} of the time, {t['lift vs base rate']} times that "
            f"class's base rate.")

    if ev["pushing_against"]:
        a = ev["pushing_against"][0]
        parts.append(f"Working against it: {a['readable']} "
                     f"({a['contribution']:+.3f}).")

    n = CLASS_ROWS.get(ev["prediction"])
    if n is not None and n < 30:
        f1 = CLASS_F1.get(ev["prediction"])
        f1_txt = f", and its cross-validated F1 is {f1:.3f}" if f1 else ""
        parts.append(f"Treat this as a suggestion rather than a decision. "
                     f"{ev['prediction']} has only {n} training examples{f1_txt}.")
    elif ev["agreement"] < 0.6:
        parts.append("The margin here is narrow, so this row is worth a human "
                     "check.")

    parts.append("(Generated without a language model. Set GROQ_API_KEY for a "
                 "written explanation.)")
    return " ".join(parts)


# ---------------------------------------------------------------- provider

def _run_agent(prompt, api_key, model, base_url, stats, metrics, trace):
    """Tool loop against an OpenAI-shaped endpoint. Drives Groq or a local
    Ollama server. The model decides which lookups it needs before writing.

    The first pass forces a tool call, so the model cannot answer from the
    prompt alone; later passes are free to stop once it has what it needs.
    Every call is recorded in trace, which the dashboard displays.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    tools = [{"type": "function",
              "function": {"name": t["name"], "description": t["description"],
                           "parameters": t["input_schema"]}} for t in TOOLS]
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]

    for step in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools, max_tokens=1000,
            tool_choice="required" if step == 0 else "auto")
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append(msg)
        for c in msg.tool_calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out = run_tool(c.function.name, args, stats, metrics)
            trace.append({"tool": c.function.name, "input": args, "output": out})
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": json.dumps(out)})

    return "The agent used its full tool budget without finishing."


# ---------------------------------------------------------------- entry point

def explain(ev, provider="offline", api_key=None, model=None,
            stats=None, metrics=None):
    """Returns (text, trace, provider_used).

    Falls back to the template on a missing key or any provider failure, so
    the Explain button in the dashboard always produces something.
    """
    stats = stats if stats is not None else load_token_stats()
    metrics = metrics if metrics is not None else load_metrics()
    trace = []

    cfg = PROVIDERS.get(provider)
    if cfg is None:
        return template_explanation(ev), trace, "offline"

    key = api_key or (os.environ.get(cfg["env"]) if cfg["env"] else None)
    if cfg["env"] and not key:
        return template_explanation(ev), trace, f"offline (no {cfg['env']})"

    try:
        text = _run_agent(evidence_prompt(ev), key, model or cfg["model"],
                          cfg["base_url"], stats, metrics, trace)
        if not text.strip():
            raise RuntimeError("empty response")
        return text, trace, provider
    except Exception as e:
        return (template_explanation(ev)
                + f"\n\n(Agent unavailable: {type(e).__name__}: {e})"), \
               trace, "offline (agent failed)"