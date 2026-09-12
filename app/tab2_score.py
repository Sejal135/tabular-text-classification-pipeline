"""Tab 2: upload a file, run the fitted pipeline, show predictions, explain one.

The explainer is on demand only. Nothing is computed for a row until the
reader selects it and presses Explain, and results are cached per row so
re-renders do not re-call the API.
"""

import os

import pandas as pd
import streamlit as st
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from common import (FEATURE_COLS, clean_frame, load_contributions,
                    load_explain_artifacts, normalize_label, read_any,
                    resolve_columns)
from src.agent import explain as run_agent
from src.explain import build_evidence


def render(pipe, le):
    """Run the whole tab: upload, validate, predict, score, and explain. Each
    step returns None on failure so the page stops cleanly rather than
    cascading errors into the next block."""
    st.write("Upload a file with columns Col1 through Col7, in any order. "
             "Header case and spacing are normalized. Extra columns are ignored. "
             "If a label column is present, scoring appears below.")

    up = st.file_uploader("Data file", type=["csv", "xls", "xlsx"])
    if up is None:
        return

    raw = _read(up)
    if raw is None:
        return

    rename, label_col = _resolve(raw)
    if rename is None:
        return

    out, pred, prob_cols = _predict(raw, rename, pipe, le)
    if out is None:
        return

    _summary(out)

    if label_col:
        _live_score(raw, out, pred, label_col)

    view = _results_table(out, prob_cols, label_col)
    _explain_panel(out, view, rename, pipe, le)


# ---------- steps ----------

def _read(up):
    """Parse the uploaded file into a dataframe, reporting what each reader
    complained about if none succeeds. Returns None on failure or 0 rows."""
    raw, reader, errors = read_any(up)
    if raw is None:
        st.error("Could not read the file with any supported reader.")
        for e in errors:
            st.caption(e)
        return None
    if raw.empty:
        st.warning(f"The file parsed correctly but contains 0 rows "
                   f"({len(raw.columns)} columns found). Nothing to score.")
        return None
    st.session_state["_reader"] = reader
    return raw


def _resolve(raw):
    """Match the file's headers to the required columns and report what was
    matched, ignored, or detected as a label. Returns (None, None) and names
    the missing columns if any required one is absent."""
    rename, missing, label_col = resolve_columns(raw)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.caption(f"Columns found in the file: {', '.join(map(str, raw.columns))}")
        return None, None

    renamed = {a: r for a, r in rename.items() if a != r}
    extra = [c for c in raw.columns if c not in rename and c != label_col]

    st.success(f"Loaded {len(raw):,} rows via {st.session_state.get('_reader')}.")
    bits = []
    if renamed:
        bits.append("matched headers: " +
                    ", ".join(f"{a} to {r}" for a, r in renamed.items()))
    if extra:
        bits.append(f"ignored extra columns: {', '.join(map(str, extra))}")
    if label_col:
        bits.append(f"label column detected: {label_col}")
    if bits:
        st.caption(" | ".join(bits))

    with st.expander("Input preview"):
        st.dataframe(raw.head(20), use_container_width=True)

    return rename, label_col


def _predict(raw, rename, pipe, le):
    """Score every row through the fitted pipeline and attach the prediction,
    the winning vote share, and one probability column per class to a copy of
    the original file."""
    try:
        X = clean_frame(raw.rename(columns=rename)[FEATURE_COLS].copy())
        proba = pipe.predict_proba(X)
        pred = le.classes_[proba.argmax(axis=1)]
    except Exception as e:
        st.error(f"Prediction failed: {type(e).__name__}: {e}")
        return None, None, None

    out = raw.copy()
    out.insert(0, "prediction", pred)
    out.insert(1, "agreement", proba.max(axis=1).round(4))
    prob_cols = []
    for j, cls in enumerate(le.classes_):
        col = f"p_{cls}"
        out[col] = proba[:, j].round(4)
        prob_cols.append(col)
    return out, pred, prob_cols


def _summary(out):
    """Show headline counts for the scored file: rows, classes predicted,
    median agreement, and the predicted class distribution."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{len(out):,}")
    c2.metric("Classes predicted", out["prediction"].nunique())
    c3.metric("Median agreement", f"{out['agreement'].median():.3f}")
    st.caption("Agreement is the share of trees that voted for the predicted "
               "class. It is a vote share, not a calibrated probability.")

    st.subheader("Predicted class counts")
    counts = (out["prediction"].value_counts()
              .rename_axis("class").reset_index(name="rows"))
    counts["share"] = (counts["rows"] / len(out) * 100).round(2)
    st.dataframe(counts, use_container_width=True, hide_index=True)


def _live_score(raw, out, pred, label_col):
    """Score the predictions against labels supplied in the uploaded file, and
    add actual and correct columns to the results table. Unparseable labels
    are excluded and counted rather than guessed at."""
    st.subheader("Live scoring against supplied labels")
    y_true = raw[label_col].map(normalize_label)
    bad = y_true.isna().sum()
    if bad:
        st.caption(f"{bad} label values could not be parsed and are excluded.")

    keep = y_true.notna().values
    if keep.sum() == 0:
        st.warning("No usable labels found in that column.")
        return

    yt, yp = y_true[keep].values, pd.Series(pred)[keep].values
    seen = sorted(set(yt))

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{(yt == yp).mean():.4f}")
    m2.metric("Macro F1",
              f"{f1_score(yt, yp, labels=seen, average='macro', zero_division=0):.4f}")
    m3.metric("Balanced accuracy", f"{balanced_accuracy_score(yt, yp):.4f}")
    st.caption("Macro F1 covers only the classes present in this file. Classes "
               "with a handful of rows produce unstable F1 values.")

    st.write("Confusion matrix")
    st.dataframe(pd.DataFrame(confusion_matrix(yt, yp, labels=seen),
                              index=[f"true {c}" for c in seen],
                              columns=[f"pred {c}" for c in seen]),
                 use_container_width=True)

    with st.expander("Per-class report"):
        st.code(classification_report(yt, yp, labels=seen,
                                      zero_division=0, digits=3))

    out.insert(2, "actual", y_true.values)
    out.insert(3, "correct", out["actual"] == out["prediction"])


def _results_table(out, prob_cols, label_col):
    """Render the filterable, selectable results table and the CSV download.
    Stores the selected row in session state for the explainer, and returns
    the filtered view so the explainer can map a selection back to a row."""
    st.subheader("Results")
    st.caption("Click a row to select it, then use the explainer below.")

    c1, c2, c3 = st.columns(3)
    only_minority = c1.checkbox("Hide Category_1 predictions", value=False)
    show_probs = c2.checkbox("Show per-class probabilities", value=False)
    errors_only = c3.checkbox("Show errors only", value=False,
                              disabled=label_col is None)

    view = out
    if only_minority:
        view = view[view["prediction"] != "Category_1"]
    if errors_only and label_col:
        view = view[~view["correct"]]

    live_probs = [c for c in prob_cols if out[c].max() > 0]
    dropped = [c for c in prob_cols if c not in live_probs]

    cols = ["prediction", "agreement"]
    if label_col:
        cols += ["actual", "correct"]
    cols += FEATURE_COLS
    if show_probs:
        cols += live_probs
        if dropped:
            st.caption(f"Hidden, never predicted: {', '.join(dropped)}")

    st.caption(f"Showing {len(view):,} of {len(out):,} rows.")
    if view.empty:
        st.info("No rows match the current filters.")
        return view

    event = st.dataframe(
        view[cols], use_container_width=True, height=560,
        on_select="rerun", selection_mode="single-row", key="results_table",
        column_config={
            "prediction": st.column_config.TextColumn(
                "prediction", pinned=True, width="small"),
            "agreement": st.column_config.NumberColumn(
                "agreement", pinned=True, format="%.3f", width="small"),
            "Col1": st.column_config.TextColumn("Col1", width="medium"),
            "Col4": st.column_config.TextColumn("Col4", width="medium"),
            "Col6": st.column_config.TextColumn("Col6", width="medium"),
            "Col3": st.column_config.NumberColumn("Col3", format="%.2f"),
        })
    st.session_state["_selection"] = list(event.selection.rows)

    st.download_button("Download predictions as CSV",
                       out.to_csv(index=False).encode(),
                       file_name="predictions.csv", mime="text/csv")
    return view


# ---------- agentic prediction explainer ----------

def _explain_panel(out, view, rename, pipe, le):
    """Show the on-demand explainer for the selected row: provider choice, the
    Explain button, and the written explanation with the evidence tables and
    agent tool calls behind it. Nothing runs until the button is pressed, and
    each result is cached per row and provider."""
    st.divider()
    st.subheader("Explain a prediction")

    if view is None or view.empty:
        return

    sel = st.session_state.get("_selection") or []
    if not sel or sel[0] >= len(view):
        st.info("Select a row in the table above to explain it.")
        return

    label = view.index[sel[0]]            # position in `out`
    row_out = out.loc[[label]]

    has_key = bool(os.environ.get("GROQ_API_KEY") or
                   st.secrets.get("GROQ_API_KEY", None))
    c1, c2 = st.columns([2, 1])
    provider = c1.radio(
        "Explainer", ["Groq agent", "Offline (no API)"],
        index=0 if has_key else 1, horizontal=True,
        help=("The agent looks up token statistics and class reliability before "
              "writing. Offline composes the same evidence into prose with no "
              "API call."))
    if provider == "Groq agent" and not has_key:
        st.caption("No GROQ_API_KEY found. This will fall back to offline.")

    go = c2.button("Explain this row", type="primary", use_container_width=True)

    st.dataframe(row_out[FEATURE_COLS], use_container_width=True, hide_index=True)

    cache_key = f"_expl_{label}_{provider}"
    if go:
        with st.spinner("Measuring feature contributions, then writing..."):
            st.session_state[cache_key] = _explain_row(
                row_out, rename, pipe, le,
                "groq" if provider == "Groq agent" else "offline")

    payload = st.session_state.get(cache_key)
    if payload is None:
        st.caption("Nothing is computed until you press the button.")
        return

    if "error" in payload:
        st.error(payload["error"])
        return

    ev, text, trace, used = payload["ev"], payload["text"], payload["trace"], payload["used"]

    st.markdown(text)
    st.caption(f"Source: {used}. Attribution: {ev['attribution_method']}.")

    c1, c2 = st.columns(2)
    with c1:
        st.write("Features supporting the prediction")
        st.dataframe(pd.DataFrame(ev["pushing_toward"]),
                     use_container_width=True, hide_index=True)
    with c2:
        st.write("Features working against it")
        if ev["pushing_against"]:
            st.dataframe(pd.DataFrame(ev["pushing_against"]),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("None.")
    st.caption("Each contribution is the drop in predicted probability when "
               "that single feature is removed. Correlated features under-count, "
               "because removing one leaves the other in place.")

    if ev["token_evidence"]:
        st.write("How these tokens behaved in training")
        st.dataframe(pd.DataFrame(ev["token_evidence"]),
                     use_container_width=True, hide_index=True)

    if trace:
        with st.expander(f"Agent tool calls ({len(trace)})"):
            for t in trace:
                st.write(f"**{t['tool']}**({t['input']})")
                st.json(t["output"], expanded=False)

    with st.expander("Model caveats for this class"):
        for c in ev["caveats"]:
            st.write(f"- {c}")


def _explain_row(row_out, rename, pipe, le, provider):
    """Measure the row's feature contributions and token evidence, then hand
    that to the agent to write up. Evidence first, prose second, never the
    other way round. Returns an error payload rather than raising."""
    try:
        contrib = load_contributions(pipe)
        stats, metrics = load_explain_artifacts()
        row_x = clean_frame(row_out.rename(columns=rename)[FEATURE_COLS].copy())
        ev = build_evidence(row_x, pipe, le, contrib, stats, metrics)
        key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
        text, trace, used = run_agent(ev, provider=provider, api_key=key,
                                      stats=stats, metrics=metrics)
        return {"ev": ev, "text": text, "trace": trace, "used": used}
    except Exception as e:
        return {"error": f"Could not explain this row: {type(e).__name__}: {e}"}