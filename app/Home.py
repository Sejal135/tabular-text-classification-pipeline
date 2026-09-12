"""EY classification challenge interactive dashboard.

Entrypoint only. Tab bodies live in tab1_analysis.py and tab2_score.py, shared
loaders in common.py, styling in style.py.

There are no functions in this file. Streamlit executes it top to bottom on
every interaction, so the order of these blocks is the page itself.

Run with:  streamlit run app/Home.py
"""

import streamlit as st

# Must be the first Streamlit call in the session, which is why it sits above
# the remaining imports rather than with them.
st.set_page_config(page_title="Transaction Classification",
                   page_icon="◧", layout="wide",
                   initial_sidebar_state="expanded")

# Deferred until after set_page_config. common.py also puts the project root on
# sys.path, which is what lets these modules import from src/.
import tab1_analysis  # noqa: E402
import tab2_score  # noqa: E402
from common import ART, load_metrics, load_model  # noqa: E402
from style import inject_css  # noqa: E402

# Global CSS: larger body text, bordered metric cards, readable captions.
inject_css()

# Load the fitted model and its scores once, cached across reruns. If the
# artifacts are missing the app stops here with an actionable message rather
# than failing later inside a tab.
try:
    pipe, le = load_model()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Could not load artifacts from {ART}: {e}")
    st.caption("Run notebooks/02_modeling.ipynb to generate them.")
    st.stop()


# ---------- sidebar: context that should never scroll out of view ----------

# Pull the headline figures out of metrics.json, defaulting rather than raising
# so an older artifacts file still renders the page.
cv = metrics.get("cv", {})
hold = metrics.get("holdout", {})
base = metrics.get("baselines", {})
dummy = base.get("dummy_macro_f1_seed42", base.get("dummy_macro_f1", 0))

# The sidebar keeps the two things most worth saying on screen at all times:
# what the model scores, and why accuracy is not that score.
with st.sidebar:
    st.markdown("### Transaction Classification")
    st.caption("Six classes, 5,899 rows, heavily imbalanced. "
               "Random forest over token features.")

    # Headline score, shown against the majority-class baseline so the number
    # is read as a gain rather than in isolation.
    st.divider()
    st.metric("Cross-validated macro F1",
              f"{cv.get('macro_f1_mean', float('nan')):.4f}",
              delta=f"{cv.get('macro_f1_mean', 0) - dummy:+.4f} vs baseline")
    st.caption(f"± {cv.get('macro_f1_std', 0):.4f} across three seeds")

    # Flagged rather than celebrated: the holdout is higher because three of
    # its five classes have one or two rows.
    st.metric("Holdout macro F1", f"{hold.get('macro_f1', float('nan')):.4f}")
    st.caption("Higher than CV, and that is an artifact. See tab 1, section 4.")

    # The metric choice, stated before anyone asks why accuracy is missing.
    st.divider()
    st.caption("**Why not accuracy**")
    st.caption(f"Predicting the majority class every time already scores "
               f"{base.get('dummy_accuracy_seed42', base.get('dummy_accuracy', 0)):.1%}. "
               f"Macro F1 is the reported metric throughout.")

    st.divider()
    st.caption("Built for the EY data challenge. All charts computed live; "
               "no pre-rendered results.")


# ---------- tabs ----------

st.title("Transaction Classification")
st.caption("Exploration, Model Selection, and Per-Row Explanation for a "
           "Six-Class accounting dataset.")

# Two tabs, each delegating to its own module. Both receive the same fitted
# pipeline, so the analysis shown and the predictions made cannot drift apart.
tab1, tab2 = st.tabs(["Analysis and Model", "Score New Data"])

# Tab 1: exploration, preprocessing, split strategy, model evaluation.
with tab1:
    tab1_analysis.render(pipe, le, metrics)

# Tab 2: upload a file, score it live, and explain any row on demand.
with tab2:
    st.header("Score New Data")
    tab2_score.render(pipe, le)