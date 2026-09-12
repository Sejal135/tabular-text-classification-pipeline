"""Interactive chart builders. Pure functions: data in, plotly figure out.

Every chart is computed live from the dataset or a saved CSV, so Tab 1 never
depends on a pre-rendered PNG. Only builders the dashboard actually calls live
here; charts used during exploration but cut from the dashboard stay in
notebooks/01_eda.ipynb.
"""

import pandas as pd
import plotly.express as px

CLASS_ORDER = ["Category_1", "Category_2", "Category_3",
               "Category_4", "Category_5", "Category_6"]
SEQ = px.colors.qualitative.Safe


def _layout(fig, height=380, title=None):
    """Apply the shared look: transparent background, faint gridlines, and
    margins that grow to fit axis labels."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="closest",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if title:                                  # None here renders as "undefined"
        fig.update_layout(title=dict(text=title, x=0, xanchor="left"))
    # automargin grows the plot area to fit tick and axis labels
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False,
                     automargin=True)
    return fig


# ---------- exploration ----------

def class_bar(counts, log_y=False):
    """Bar chart of rows per class. Log scale makes the rare classes visible."""
    d = counts.rename_axis("class").reset_index(name="rows")
    d["share %"] = (d["rows"] / d["rows"].sum() * 100).round(2)
    fig = px.bar(d, x="class", y="rows", color="class",
                 color_discrete_sequence=SEQ,
                 hover_data={"share %": True, "class": False},
                 text="rows")
    fig.update_traces(textposition="outside")
    if log_y:
        fig.update_yaxes(type="log", title="rows (log scale)")
    fig.update_layout(showlegend=False)
    return _layout(fig)


def coverage_table(df, cols, threshold=0.95):
    """How many distinct values each column needs to cover a share of rows.
    A high number means rare-value grouping cannot work for that column."""
    rows = []
    for c in cols:
        vc = df[c].value_counts()
        cum = vc.cumsum() / len(df)
        rows.append({
            "column": c,
            "distinct values": len(vc),
            f"needed for {threshold:.0%}": int((cum < threshold).sum() + 1),
            "appear once": int((vc == 1).sum()),
        })
    return pd.DataFrame(rows).set_index("column")


def cramers_heatmap(V):
    """Heatmap of pairwise Cramer's V, showing which columns say the same
    thing and which actually track the label."""
    fig = px.imshow(V, text_auto=".3f", zmin=0, zmax=1,
                    color_continuous_scale="Viridis", aspect="auto")
    fig.update_xaxes(side="bottom")
    return _layout(fig, 460)


def amount_hist(df, use_log, nbins=80):
    """Histogram of Col3, raw or signed log, showing why the transform is
    needed on a distribution with skew 8.39."""
    col = "col3_signed_log" if use_log else "Col3"
    fig = px.histogram(df, x=col, nbins=nbins, color_discrete_sequence=[SEQ[0]])
    fig.update_xaxes(title="signed log amount" if use_log else "amount")
    return _layout(fig)


def amount_box_by_class(df, use_log):
    """Box plot of Col3 split by class. Shows spread, outliers, and how much
    the amount alone separates one class from another."""
    col = "col3_signed_log" if use_log else "Col3"
    order = [c for c in CLASS_ORDER if c in set(df["label"])]
    fig = px.box(df, x=col, y="label", color="label", points="outliers",
                 category_orders={"label": order}, color_discrete_sequence=SEQ)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="signed log amount" if use_log else "amount")
    return _layout(fig, 420)


# ---------- preprocessing ----------

def branch_pie(branch):
    """Donut of how the model's features split across pipeline branches, so
    the dominance of the token vectorizers is visible at a glance."""
    d = branch.rename_axis("branch").reset_index(name="features")
    fig = px.pie(d, names="branch", values="features", hole=0.45,
                 color_discrete_sequence=SEQ)
    fig.update_traces(textinfo="label+value")
    return _layout(fig, 380)


# ---------- evaluation ----------

def seed_bar(series, mean=None):
    """Macro F1 per random seed with the mean marked, showing how much the
    score moves purely from reshuffling the folds."""
    d = series.rename_axis("seed").reset_index(name="macro F1")
    d["seed"] = d["seed"].astype(str)
    fig = px.bar(d, x="seed", y="macro F1", text="macro F1",
                 color_discrete_sequence=[SEQ[2]])
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    if mean is not None:
        fig.add_hline(y=mean, line_dash="dash", line_color="crimson",
                      annotation_text=f"mean {mean:.4f}")
    fig.update_yaxes(range=[0, max(1.0, d["macro F1"].max() * 1.15)])
    return _layout(fig, 340)