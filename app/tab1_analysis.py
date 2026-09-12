"""Tab 1: exploration, preprocessing, split strategy, model evaluation.

Everything here is computed live and rendered with plotly. Controls let the
viewer flip between before and after views rather than reading two static
tables side by side.

Every st.plotly_chart call carries an explicit key. Streamlit identifies
elements by type plus parameters, so two charts built from identical figures
collide without one.
"""

import pandas as pd
import streamlit as st

import charts as ch
from common import (ART, FEATURE_COLS, load_clean, load_holdout_labels,
                    load_raw, show_csv)


def render(pipe, le, metrics):
    """Entry point. Loads the data once, then delegates to one function per
    required section."""
    try:
        raw_df, clean_df = load_raw(), load_clean()
        have_data = True
    except Exception as e:
        st.warning(f"Raw data not loadable, showing saved artifacts only: {e}")
        raw_df = clean_df = None
        have_data = False

    s1, s2, s3, s4 = st.tabs([
        "1. Exploration", "2. Cleaning and Preprocessing",
        "3. Split Strategy", "4. Model and Evaluation"])

    with s1:
        _exploration(raw_df, clean_df, have_data)
    with s2:
        _preprocessing(raw_df, clean_df, have_data, pipe)
    with s3:
        _split(clean_df, have_data)
    with s4:
        _evaluation(metrics)


def _ba(key, labels=("Before", "After")):
    """Before/after switch. Returns True when the second label is selected."""
    return st.radio("view", labels, horizontal=True, key=key,
                    label_visibility="collapsed") == labels[1]


# ---------------------------------------------------------------- 1

def _exploration(raw_df, clean_df, have_data):
    """Distributions, cardinality, correlations, outliers, leakage."""
    if not have_data:
        st.info("Raw data unavailable. Charts in this section need "
                "data/raw/Challenge_Data.xls.")
        return

    st.subheader("What the Data is")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(raw_df):,}")
    c2.metric("Columns", len(raw_df.columns))
    c3.metric("Classes", clean_df["label"].nunique())
    c4.metric("Majority share",
              f"{clean_df['label'].value_counts(normalize=True).iloc[0]:.1%}")

    st.write("Text columns are anonymized into WordN tokens, so no pretrained "
             "embedding or language model can interpret them. Every feature here "
             "is statistical.")

    st.dataframe(pd.DataFrame({
        "dtype": raw_df.dtypes.astype(str),
        "unique": raw_df.nunique(),
        "nulls": raw_df.isna().sum(),
        "null %": (raw_df.isna().mean() * 100).round(2),
    }), use_container_width=True)
    st.caption("Col2 reads as object because it mixes integers and strings. Col4 "
               "is the only column with missing values, 153 of them.")

    st.divider()
    st.subheader("Class Imbalance")
    log_y = st.toggle("Log scale", value=False, key="imb_log",
                      help="Linear scale makes Category_3 through 6 invisible. "
                           "That is the point, but log makes them readable.")
    vc = clean_df["label"].value_counts()
    st.plotly_chart(ch.class_bar(vc, log_y), use_container_width=True,
                    key="ex_class_bar")
    st.warning(f"Predicting the majority class for every row scores "
               f"{vc.iloc[0] / len(clean_df):.1%} accuracy. That is why accuracy "
               f"is not the headline metric. Category_5 has {int(vc.min())} rows "
               f"in the entire dataset.")

    # the number is the whole argument here, so the curve and its controls are
    # in the notebook rather than on screen
    st.divider()
    st.subheader("High Cardinality")
    st.dataframe(ch.coverage_table(clean_df, ["Col1", "Col4", "Col6"], 0.95),
                 use_container_width=True)
    st.caption("How many distinct values are needed to cover 95% of rows. Col4 "
               "needs roughly 1,900 of its 2,032, so bucketing rare values into "
               "an Other category is not a grouping at all. That is what pushed "
               "these three columns to token features instead. Col1 reaches 95% "
               "at roughly 130 values and Col6 at roughly 70.")

    st.divider()
    st.subheader("Feature Strength, and Why 1 Measure Misleads")
    st.write("Mutual information against the label")
    show_csv("mutual_information.csv")
    st.caption("This ranking is almost exactly the ranking by distinct-value "
               "count, which is the tell: the measure is rewarding cardinality, "
               "not signal. Col2 tops it at 99% of the label's entropy while "
               "predicting nothing new, because 88% of its values appear exactly "
               "once.")

    V = show_csv("cramers_v.csv")
    if V is not None:
        st.write("Cramer's V, bias corrected")
        st.plotly_chart(ch.cramers_heatmap(V), use_container_width=True,
                        key="ex_cramers")
        st.caption("Bias correction penalizes a column for having thousands of "
                   "values. Col2 drops from first to fourth at 0.426. Col4 is "
                   "the strongest real feature at 0.78. Col6 against Col7 is "
                   "0.961, meaning Col7 is a coarse version of Col6. Two "
                   "measures disagreeing on the same column is the evidence for "
                   "dropping raw Col2.")

    st.divider()
    st.subheader("Amount Distribution and Outliers")
    use_log = st.toggle("Signed log transform", value=False, key="amt_log",
                        help="Raw skew is 8.39. Signed rather than plain log "
                             "because 69 values are negative.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(ch.amount_hist(clean_df, use_log),
                        use_container_width=True, key="ex_amount_hist")
    with c2:
        st.plotly_chart(ch.amount_box_by_class(clean_df, use_log),
                        use_container_width=True, key="ex_amount_box")
    st.caption("Flip the transform on and off. Raw, everything collapses into one "
               "bar and the document totals sit alone at 8.8 million. "
               "Transformed, three modes appear: the main body, a spike from "
               "document totals, and a left cluster of large credits. "
               "Category_5's two rows sit alone at the far left, which looks like "
               "a perfect separator and is not one at n=2.")

    with st.expander("Amount anomalies found, and what was done about them"):
        st.markdown(
            "- 8,799,523.59 appears 59 times under a single Col2 value, and "
            "2,829,197.98 appears 35 times under one. These are document totals "
            "copied onto every line item, roughly 94 rows. Reported as a data "
            "quality finding, not turned into a feature, because every affected "
            "row is already Category_1.\n"
            "- 269,111.58 appears 34 times positive and 35 times negative, each "
            "across different documents. That is an accrual and its reversal, "
            "which is normal accounting. 44,473.31 shows the same pattern at +19 "
            "and -15.\n"
            "- 35.8% of amounts are multiples of ten, and the top repeated values "
            "are round numbers, so an is_round_amount flag was built.\n"
            "- No sentinel values. Two zeros, nothing at -1 or 9999.")

    st.divider()
    st.subheader("Leakage Check")
    pure = show_csv("pure_value_check.csv", index_col=None, height=260)
    if pure is not None and "class" in pure.columns:
        minority = pure[pure["class"] != "Category_1"]
        st.caption(f"{len(pure)} pure values at 5 rows or more, of which "
                   f"{len(minority)} point at a class other than Category_1.")
    st.caption("None of these is derived from the label. In a tax dataset a value "
               "that always maps to one class is usually the business rule "
               "itself. The useful finding was shared tokens: Word575 appears in "
               "three separate values totalling 114 rows, all Category_2, and "
               "Word905 covers 11 of the 23 Category_3 rows. That is the evidence "
               "for tokenizing rather than one-hot encoding whole values.")

    with st.expander("Ideas Tested and Rejected"):
        st.markdown(
            "- Col4 missingness as a feature. Missing rows are 91.5% Category_1 "
            "against 88.3% for present rows. Noise at n=153.\n"
            "- Deduplication. 1,215 rows sit in 535 exact-duplicate groups. In "
            "accounting data two identical line items can both be genuine, and "
            "removing them would have pushed Category_1 from 88.4% to 89.2%. All "
            "5,899 rows kept; the train/test leakage risk is handled at the split "
            "instead.\n"
            "- Col1 positional tokens. The first token gives 277 distinct values "
            "from 313, so there is no hierarchy. Col6 was the opposite at 24 from "
            "196.\n"
            "- Col5 as a feature. Only 17 distinct dates exist and class shares "
            "hold between 86% and 91% Category_1 across all of 2018, so there is "
            "no drift to model.")
        show_csv("col4_nullness_by_class.csv")


# ---------------------------------------------------------------- 2

def _preprocessing(raw_df, clean_df, have_data, pipe):
    """Before and after views for every transformation applied."""
    if have_data:
        st.subheader("Labels")
        after = _ba("lbl_ba", ("As delivered", "After normalization"))
        if after:
            st.dataframe(clean_df["label"].value_counts()
                         .rename_axis("class").reset_index(name="rows"),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(ch.class_bar(clean_df["label"].value_counts(), True),
                            use_container_width=True, key="pp_class_bar")
            st.success("Six classes, 5,899 rows, nothing unparsed. One rule "
                       "handles all four failure modes: extract the digit.")
        else:
            st.dataframe(raw_df["ClassificationLabel"].value_counts(dropna=False)
                         .rename_axis("raw value").reset_index(name="rows"),
                         use_container_width=True, hide_index=True)
            st.error("Eleven distinct strings for six classes: inconsistent "
                     "casing, a stray space, a space before the underscore, and "
                     "the typo Categry_6.")

        st.divider()
        st.subheader("Text Columns: Cleaning was a no-op")
        cats = ["Col1", "Col2", "Col4", "Col6", "Col7"]
        st.dataframe(pd.DataFrame({
            "as delivered": {c: raw_df[c].nunique() for c in cats},
            "whitespace trimmed": {c: clean_df[c].nunique() for c in cats},
            "also lowercased": {c: clean_df[c].astype(str).str.lower().nunique()
                                for c in cats},
        }), use_container_width=True)
        st.caption("Identical across all three columns, so nothing was actually "
                   "fixed. Every formatting problem in this file was in the label "
                   "column. The cleaning functions are kept as a guard for unseen "
                   "data, and the no-op is reported rather than dressed up.")

        st.divider()
        st.subheader("Engineered Features")
        after = _ba("feat_ba", ("Columns as delivered", "With derived columns"))
        derived = ["col2_is_numeric", "col2_len", "col2_shape_coarse",
                   "col3_signed_log", "is_round_amount", "is_negative",
                   "col6_first_token"]
        if after:
            st.dataframe(clean_df[["Col2", "Col3", "Col6"] + derived].head(15),
                         use_container_width=True)
            st.caption("Seven derived columns, each computed from a single row's "
                       "own values, so none of them can leak across the "
                       "train/test boundary.")
        else:
            st.dataframe(raw_df[["Col2", "Col3", "Col6"]].head(15),
                         use_container_width=True)
            st.caption("The same rows before derivation. Col2 is a reference code "
                       "with 4,818 distinct values, useless as a category.")

        st.divider()
        st.subheader("2 Features worth defending")
        pick = st.selectbox("Feature", ["col2_is_numeric", "col6_first_token"],
                            key="feat_pick")

        s = clean_df[pick]
        if s.nunique() > 12:
            keep = s.value_counts()[lambda x: x >= 30].index
            s = s.where(s.isin(keep))
        st.dataframe(pd.crosstab(s, clean_df["label"]), use_container_width=True)
        st.dataframe(pd.crosstab(s, clean_df["label"], normalize="index")
                     .mul(100).round(2), use_container_width=True)

        notes = {
            "col2_is_numeric":
                "Reference codes made only of digits are nine times more likely "
                "to be Category_2: 16.65% against 1.86%. Computed with a regex, "
                "not the pandas dtype, because the dtype only exists as an "
                "artifact of how Excel parsed the file and vanishes on a CSV "
                "round trip. The two agree on all 5,899 rows.",
            "col6_first_token":
                "196 Col6 values collapse into 24 first tokens, and the rare "
                "classes concentrate in two of them. Word576 holds 11 of the 16 "
                "Category_4 rows and Word571 holds 9 of the 12 Category_6 rows. "
                "Nothing else tested locates those classes at all. Values under "
                "30 rows are hidden here.",
        }
        st.caption("Counts above, class share as a percentage below. "
                   + notes.get(pick, ""))

    st.divider()
    st.subheader("The Fitted Pipeline")
    st.markdown("One object takes raw Col1 through Col7 and returns a prediction. "
                "Cleaning and feature derivation are steps inside it, not "
                "something a caller has to remember, which is what makes Tab 2 "
                "work from a bare CSV.")
    try:
        names = pipe.named_steps["features"].get_feature_names_out()
        branch = pd.Series([n.split("__")[0] for n in names]).value_counts()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Input columns", len(FEATURE_COLS))
            st.metric("Model features", f"{len(names):,}")
            st.caption("min_df prunes tokens seen once, which is why 2,032 Col4 "
                       "values become far fewer token features.")
        with c2:
            st.plotly_chart(ch.branch_pie(branch), use_container_width=True,
                            key="pp_branch_pie")
    except Exception as e:
        st.caption(f"Could not introspect the pipeline: {e}")

    st.markdown("""
| Column | Treatment | Why |
|---|---|---|
| Col1, Col4, Col6 | TF-IDF over whitespace tokens, min_df 5 / 5 / 3 | Values share tokens. Encoding whole values makes the model learn Word575 three separate times from 56, 50, and 8 rows instead of once from 114. |
| Col6 | first token, one-hot, on top of TF-IDF | TF-IDF discards position, and position is what carries the hierarchy. |
| Col2 | dropped as a value, kept as is_numeric, length, and shape mask | 88% of its values appear exactly once. As a category it memorizes rows and generalizes to nothing. |
| Col3 | signed log, is_round_amount, is_negative, median imputer, scaler | Skew 8.39 and 69 negative values, so plain log fails. |
| Col5 | dropped | Only 17 distinct dates, class shares hold between 86% and 91% Category_1 across 2018 so there is no drift, it has the lowest mutual information of any column, and removing it moved the score by 0.005 against seed noise of 0.05. |
| Col7 | one-hot | Kept, though Cramer's V of 0.961 against Col6 says it is largely redundant. |
    """)


# ---------------------------------------------------------------- 3

def _split(clean_df, have_data):
    """How the Data was Partitioned, and the Limitation that follows from it."""
    st.subheader("How the Data was Partitioned")
    st.markdown("Two problems drove this design. Category_5 has two rows in the "
                "entire dataset, so it cannot be split at all. And 1,215 rows are "
                "exact duplicates of each other, so a naive split would put a row "
                "in training and its twin in the holdout, letting the model score "
                "well from memory rather than skill.")
    st.markdown("""
1. Set Category_5 aside. Two rows, held out of the split entirely.
2. Group-aware stratified holdout on the remaining 5,897, using StratifiedGroupKFold with a fixed seed. Rows with identical Col1 through Col7 share a group key and always move together, so no duplicate straddles the boundary. Verified: zero groups appear on both sides.
3. Return Category_5 to the modeling set. The holdout therefore contains no Category_5, which was expected and is stated rather than hidden.
4. Cross-validation on the modeling set: three folds, group-aware, stratified, with both Category_5 rows forced into every training fold and never into validation.
    """)

    if have_data:
        hl = load_holdout_labels()
        if hl is not None:
            full = clean_df["label"].value_counts()
            hold = hl["label"].value_counts()
            tbl = pd.DataFrame({
                "full dataset": full,
                "modeling set": full.sub(hold, fill_value=0).astype(int),
                "holdout": hold,
            }).fillna(0).astype(int)

            as_pct = st.toggle("Show as % of each class", value=False,
                               key="split_pct")
            st.dataframe(tbl.div(tbl["full dataset"], axis=0).mul(100).round(1)
                         if as_pct else tbl, use_container_width=True)
            st.caption(f"Holdout is {hold.sum()} rows, "
                       f"{hold.sum() / len(clean_df):.1%} of the data. It landed "
                       f"on 10.0% rather than being forced there, because whole "
                       f"groups move together and cannot be cut.")

    st.error("The limitation, stated up front. Category_6 has one row in the "
             "holdout and Categories 3 and 4 have two each. An F1 score on one "
             "row is not a measurement. The holdout measures Category_1 and "
             "Category_2; cross-validation on 21, 14, and 11 rows is the real "
             "evidence for the rare classes. Category_5 is never validated "
             "anywhere, so no recall figure for it exists.")

    st.subheader("Why 3 folds and not 5")
    st.markdown("Stratified k-fold needs at least k members per class. Category_5 "
                "has two and Category_6 has eleven, so five folds would leave "
                "some validation folds empty for those classes. Three folds gives "
                "each validation fold roughly seven Category_3 rows, four to five "
                "Category_4, and three to four Category_6. Fold-to-fold variance "
                "on those classes is large, so results are reported as a range "
                "across seeds rather than a single averaged number.")


# ---------------------------------------------------------------- 4

def _evaluation(metrics):
    """Model comparison, tuning, cross-validation, and confusion matrices."""
    cvm = metrics.get("cv", {})
    hom = metrics.get("holdout", {})
    base = metrics.get("baselines", {})
    dummy = base.get("dummy_macro_f1_seed42", base.get("dummy_macro_f1", 0))

    st.subheader("Headline")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV macro F1", f"{cvm.get('macro_f1_mean', float('nan')):.4f}",
              delta=f"{cvm.get('macro_f1_mean', 0) - dummy:+.4f} vs baseline")
    c2.metric("CV std across seeds", f"{cvm.get('macro_f1_std', float('nan')):.4f}")
    c3.metric("Holdout macro F1", f"{hom.get('macro_f1', float('nan')):.4f}")
    c4.metric("Holdout accuracy", f"{hom.get('accuracy', float('nan')):.4f}")

    st.info("The holdout scores higher than cross-validation, and that is not "
            "good news. Three of the five classes in the holdout have one or two "
            "rows, and the model happened to get all five right, which produces "
            "F1 values of 1.000 on no evidence. The trustworthy comparison is the "
            "classes with real sample sizes: Category_1 scores 0.962 on holdout "
            "against 0.965 in CV, accuracy is 0.934 against 0.938, and the "
            "Category_1 to Category_2 error rate is 5.6% against 5.3%. Those "
            "agree closely, which is the actual evidence that the model "
            "generalizes. The cross-validation figure is the honest headline.")

    st.divider()
    st.subheader("Algorithms Compared")
    comp = show_csv("model_comparison.csv", index_col=None)
    if comp is None:
        st.caption("Run the model_comparison cell in 02_modeling.ipynb to "
                   "generate this table.")
    st.caption("All scored on the same group-aware splitter at seed 42. The dummy "
               "row is the wall: predicting the majority class every time. Note "
               "that macro F1 and balanced accuracy disagree on the ordering, "
               "because balanced accuracy counts only recall while macro F1 also "
               "counts precision. Logistic regression and the weighted two-stage "
               "model catch more rare-class rows and pay for it in false "
               "positives.")

    st.divider()
    st.subheader("Class Weighting Mattered more than Hyperparameters")
    st.markdown("Turning on balanced class weights moved macro F1 from 0.594 to "
                "0.755. Five weightings were compared and plain balanced led in "
                "every fold. Capping the weights at 20 scored 0.706, and "
                "balanced_subsample scored 0.582, the latter because it "
                "recomputes weights on each tree's bootstrap sample and "
                "Category_5's two rows make those weights swing wildly.")

    st.divider()
    st.subheader("Hyperparameter Search")
    st.markdown("Forty candidates, three folds each, scored on the same macro F1 "
                "as everything else. Search best was 0.7588 against 0.7554 "
                "untuned, a gain of 0.0034 on a metric whose fold spread is 0.11. "
                "Re-tested across three seeds, the tuned configuration averaged "
                "0.7452 and the untuned 0.7325, so the gain is real but small. "
                "Only min_samples_leaf showed a genuine trend, with 1 best; every "
                "other parameter was a flat cloud. The bottleneck is data, not "
                "hyperparameters. Category_1, Category_2, and Category_3 already "
                "score 0.965, 0.768, and 0.894. Macro F1 is held down by "
                "Category_4 and Category_6, which have 14 and 11 training rows. "
                "No forest setting creates examples.")

    p = ART / "cv_results.csv"
    if p.exists():
        with st.expander("Full search results, all 40 candidates"):
            st.dataframe(pd.read_csv(p), use_container_width=True, height=300)

    st.divider()
    st.subheader("Cross-validation, Three Seeds")
    by_seed = cvm.get("macro_f1_by_seed", {})
    if by_seed:
        s = pd.Series(by_seed).astype(float)
        st.plotly_chart(ch.seed_bar(s, cvm.get("macro_f1_mean")),
                        use_container_width=True, key="ev_seed_bar")
        st.caption("Reshuffling the folds moves the score by more than any "
                   "modeling choice tested. That spread is why single-seed "
                   "comparisons were not trusted.")

    st.divider()
    st.subheader("Confusion Matrices")
    c1, c2 = st.columns(2)
    which = c1.radio("Source", ["Cross-validation", "Holdout"], horizontal=True,
                     key="cm_src")
    norm = c2.toggle("Normalize by true class", value=False, key="cm_norm")

    fname = ("confusion_matrix_cv.csv" if which == "Cross-validation"
             else "confusion_matrix_holdout.csv")
    fp = ART / fname
    if fp.exists():
        cm = pd.read_csv(fp, index_col=0)
        if norm:
            cm = cm.div(cm.sum(axis=1).replace(0, pd.NA), axis=0).mul(100).round(1)
        st.dataframe(cm, use_container_width=True)
    else:
        st.caption(f"Not found: artifacts/{fname}")

    st.caption("Rows are true classes, columns are predictions. Normalized, the "
               "diagonal reads as per-class recall. The dominant error in both "
               "sources is Category_1 predicted as Category_2, at 5.3% and 5.6%, "
               "which is where the most rows are lost. Category_3 quietly absorbs "
               "errors from Category_4 and Category_6, which is why its recall is "
               "perfect while its precision is 0.808.")

    rp = ART / "classification_report_cv.txt"
    if rp.exists():
        with st.expander("Classification report, cross-validation"):
            st.code(rp.read_text())

    st.divider()
    st.subheader("Rejected after Testing")
    rej = metrics.get("rejected", {})
    if rej:
        st.dataframe(pd.Series(rej).astype(str).rename("result")
                     .rename_axis("approach").reset_index(),
                     use_container_width=True, hide_index=True)
    st.markdown("Threshold tuning is worth singling out. Tuning per-class "
                "probability multipliers on out-of-fold predictions raised macro "
                "F1 from 0.7345 to 0.7601, a clear 0.026 gain. Freezing those "
                "weights and applying them to two other fold arrangements gave "
                "-0.0167 and +0.0111, roughly zero. The weights had memorized one "
                "fold arrangement rather than learning anything, so the final "
                "model uses plain argmax with no thresholds.")

    st.divider()
    st.subheader("Category_5, and What cannot be claimed")
    st.markdown("Two rows in the entire dataset. Leave-one-out training on one "
                "and testing on the other: neither was predicted correctly, but "
                "the model assigned Category_5 a probability of about 0.16 both "
                "times, roughly 400 times its 0.04% base rate, ranking it third "
                "of six. A single example does move the model, just not enough to "
                "win. Both rows would flip with a boost of about 3x, and that "
                "boost was deliberately not applied, because fitting a decision "
                "rule to two rows is exactly the mistake the threshold-tuning "
                "result already demonstrated.")