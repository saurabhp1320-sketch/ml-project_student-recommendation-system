from __future__ import annotations

import csv
import io
import warnings
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from recommender import RecConfig, UniversalRecommender, auto_config


# ---------------------------------------------------------------------
# LearnWise Universal CSV Studio + Personalised Recommender
# Requirements: streamlit, pandas, numpy, scipy, scikit-learn
# Run: streamlit run app.py
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="LearnWise | Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "4.0"


# ------------------------------- Styling -------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7fc; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151b35 0%, #25345d 100%);
    }
    [data-testid="stSidebar"] * { color: #f5f7ff; }
    .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.035em; }
    .hero {
        padding: 30px 34px; border-radius: 22px; color: white;
        background: linear-gradient(115deg, #202653, #5145b8 58%, #21829b);
        margin-bottom: 22px; box-shadow: 0 14px 38px rgba(31, 42, 93, .15);
    }
    .hero .eyebrow { color: #dce4ff; font-size: .72rem; font-weight: 800; letter-spacing: .15em; }
    .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3rem); margin: .45rem 0; }
    .hero p { color: #edf1ff; margin: 0; max-width: 850px; }
    .metric-card {
        background: white; border: 1px solid #e7eaf2; border-radius: 16px;
        padding: 17px 19px; min-height: 104px; box-shadow: 0 5px 18px rgba(25, 35, 75, .04);
    }
    .metric-label { color: #6d7690; font-size: .78rem; font-weight: 700; }
    .metric-value { color: #20294b; font: 800 1.8rem 'Manrope', sans-serif; margin-top: 7px; }
    div[data-testid="stDataFrame"] { border: 1px solid #e4e8f1; border-radius: 12px; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 10px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------ CSV engine ------------------------------

DELIMITER_OPTIONS = {
    "Auto detect": None,
    "Comma (,)": ",",
    "Semicolon (;)": ";",
    "Tab": "\t",
    "Pipe (|)": "|",
}


def _delimiter_score(text: str, delimiter: str) -> tuple[int, float]:
    """Score a delimiter by how consistently it splits the first non-empty rows."""
    try:
        rows = [
            row for row in csv.reader(io.StringIO(text[:200_000]), delimiter=delimiter)
            if row and any(str(cell).strip() for cell in row)
        ][:30]
        if len(rows) < 2:
            return (0, 0.0)
        widths = [len(row) for row in rows]
        width = max(set(widths), key=widths.count)
        consistency = sum(n == width for n in widths) / len(widths)
        return (width if consistency >= 0.5 else 0, consistency)
    except Exception:
        return (0, 0.0)


def _choose_delimiter(text: str) -> str:
    scores = [(d, _delimiter_score(text, d)) for d in [",", ";", "\t", "|"]]
    scores.sort(key=lambda item: (item[1][0], item[1][1]), reverse=True)
    return scores[0][0] if scores and scores[0][1][0] > 1 else ","


def load_csv(
    uploaded_file,
    delimiter_choice: str = "Auto detect",
    header_row: bool = True,
    skip_bad_lines: bool = False,
) -> tuple[pd.DataFrame, str, str, str]:
    """Read a CSV using several common encodings and separator strategies."""
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("The selected file is empty.")

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    errors = []
    explicit_delimiter = DELIMITER_OPTIONS.get(delimiter_choice)

    for encoding in encodings:
        try:
            sample = raw[:250_000].decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue

        delimiter = explicit_delimiter or _choose_delimiter(sample)
        separators = [delimiter]
        if explicit_delimiter is None:
            separators = list(dict.fromkeys([delimiter, ",", ";", "\t", "|"]))

        for sep in separators:
            try:
                frame = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=encoding,
                    sep=sep,
                    engine="python",
                    header=0 if header_row else None,
                    on_bad_lines="skip" if skip_bad_lines else "error",
                    skipinitialspace=True,
                    low_memory=False,
                )
                if frame.shape[1] == 1 and explicit_delimiter is None:
                    continue
                if frame.empty and frame.shape[1] == 0:
                    continue
                return frame, encoding, ("TAB" if sep == "\t" else sep), "standard"
            except Exception as exc:
                errors.append(f"{encoding}/{repr(sep)}: {exc}")

        # Last attempt: pandas delimiter inference.
        if explicit_delimiter is None:
            try:
                frame = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=encoding,
                    sep=None,
                    engine="python",
                    header=0 if header_row else None,
                    on_bad_lines="skip" if skip_bad_lines else "error",
                    skipinitialspace=True,
                )
                if frame.shape[1] > 1:
                    return frame, encoding, "Auto inferred", "inferred"
            except Exception as exc:
                errors.append(f"{encoding}/auto: {exc}")

    details = "\n".join(errors[-4:])
    raise ValueError(
        "The file could not be parsed as a consistent CSV. "
        "Try selecting the separator manually or enabling 'Skip malformed rows'.\n\n"
        + details
    )


def clean_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """Create readable, unique column names without changing cell values."""
    result = frame.copy()
    names = []
    seen = {}
    for index, value in enumerate(result.columns):
        name = str(value).replace("\ufeff", "").strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"Column_{index + 1}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        names.append(name)
    result.columns = names
    return result


def infer_types(frame: pd.DataFrame):
    """Infer numeric/date fields carefully while retaining original text otherwise."""
    data = frame.copy()
    numeric_cols, date_cols = [], []

    for col in data.columns:
        series = data[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        values = series.dropna()
        if values.empty:
            continue

        text = values.astype(str).str.strip()
        cleaned = (
            text.str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(r"^\$", "", regex=True)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")

        # Only convert if nearly all non-empty values are genuinely numeric.
        if numeric.notna().mean() >= 0.95:
            full = (
                series.astype("string").str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(r"^\$", "", regex=True)
            )
            data[col] = pd.to_numeric(full, errors="coerce")
            numeric_cols.append(col)
            continue

        # Avoid interpreting arbitrary identifiers as dates.
        if len(values) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(values, errors="coerce")
            if parsed.notna().mean() >= 0.95:
                data[col] = pd.to_datetime(series, errors="coerce")
                date_cols.append(col)

    categorical_cols = [
        col for col in data.columns
        if col not in numeric_cols and col not in date_cols
    ]
    return data, numeric_cols, categorical_cols, date_cols


def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def safe_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def category_counts(series: pd.Series, limit: int = 20) -> pd.Series:
    return (
        series.astype("string")
        .fillna("(Missing)")
        .value_counts()
        .head(limit)
    )


# --------------------------------- Header ---------------------------------

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">LEARNWISE · PERSONALISED LEARNING RECOMMENDER</div>
      <h1>Upload any CSV. Get recommendations that fit the learner.</h1>
      <p>LearnWise reads your own columns - interests, preferences, performance and behaviour -
      and recommends the topics, resources or formats that suit each student. Explore, chart and
      quality-check the data too.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------- Samples --------------------------------

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
SAMPLE_FILES = {
    "None": None,
    "Students (500 learners)": "students_sample.csv",
    "Course catalogue (35 courses)": "course_catalog.csv",
}


class SampleFile:
    """Minimal stand-in for an uploaded file so bundled samples use the same loader."""

    def __init__(self, path: Path):
        self.name = path.name
        self._bytes = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._bytes


# --------------------------------- Sidebar --------------------------------

with st.sidebar:
    st.markdown("# 🎓 LearnWise")
    st.caption("PERSONALISED LEARNING RECOMMENDER")
    st.caption(f"VERSION {APP_VERSION}")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Recommendations",
            "Overview",
            "Data explorer",
            "Visual analytics",
            "Column insights",
            "Data quality",
        ],
    )

    st.divider()
    st.markdown("### Import your dataset")
    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv", "txt"],
        help="CSV files with comma, semicolon, tab or pipe separators are supported.",
        key="learnwise_csv_upload",
    )
    delimiter_choice = st.selectbox(
        "Separator",
        list(DELIMITER_OPTIONS.keys()),
        index=0,
        help="Use Auto detect first. Choose a separator manually if columns are not split correctly.",
    )
    sample_choice = st.selectbox(
        "…or try a sample dataset",
        list(SAMPLE_FILES.keys()),
        index=0,
        help="Bundled demo files from the sample_data folder.",
    )
    header_row = st.checkbox("First row contains column names", value=True)
    skip_bad_lines = st.checkbox(
        "Skip malformed rows",
        value=False,
        help="Use only if the file has inconsistent rows. Some records may be skipped.",
    )
    st.divider()
    st.caption("Supported: common CSV encodings and flexible column names.")


if uploaded is None and SAMPLE_FILES.get(sample_choice):
    sample_path = SAMPLE_DIR / SAMPLE_FILES[sample_choice]
    if sample_path.exists():
        uploaded = SampleFile(sample_path)
    else:
        st.warning(f"Sample file not found: {sample_path}")

if uploaded is None:
    st.info("Upload a CSV file using the import panel in the sidebar - or pick a sample dataset - to start.")
    st.markdown(
        """
        **Any CSV works** - the recommender detects interests, preferences, performance and
        "what to recommend" columns automatically. For example:
        - Student records (interests, study hours, scores, preferred formats)
        - Course / resource catalogues (titles, tags, levels, ratings, descriptions)
        - Student or employee records
        - Sales, orders, products and customer data
        - Survey responses and research data
        - Weather, finance, inventory or operational data

        No fixed column names or predefined template is required.
        """
    )
    st.stop()


# --------------------------------- Load data ------------------------------

try:
    with st.spinner("Reading and preparing your CSV..."):
        raw_df, used_encoding, used_separator, parse_mode = load_csv(
            uploaded,
            delimiter_choice=delimiter_choice,
            header_row=header_row,
            skip_bad_lines=skip_bad_lines,
        )
except Exception as exc:
    st.error("CSV import failed.")
    st.exception(exc)
    st.info(
        "Check that the file is a CSV, then try Auto detect or select its separator "
        "manually in the sidebar."
    )
    st.stop()

df = clean_headers(raw_df)
df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

if df.empty or len(df.columns) == 0:
    st.error("The file was read, but it contains no usable rows or columns.")
    st.stop()

df, numeric_cols, categorical_cols, date_cols = infer_types(df)
all_columns = df.columns.tolist()

with st.expander("Import details", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"**File:** {uploaded.name}")
    c2.write(f"**Encoding:** {used_encoding}")
    c3.write(f"**Separator:** {used_separator}")
    c4.write(f"**Rows loaded:** {len(df):,}")
    st.caption(
        "If 'Skip malformed rows' is enabled, inconsistent records may have been skipped."
    )

st.caption(
    f"**{uploaded.name}**  ·  {len(df):,} rows  ·  {len(all_columns):,} columns  ·  "
    f"{len(numeric_cols)} numeric  ·  {len(categorical_cols)} text/category  ·  "
    f"{len(date_cols)} date"
)


# ----------------------------- Recommendation page ----------------------------

@st.cache_resource(show_spinner="Building the recommender from your CSV...")
def build_engine(_df, _cfg, _numeric, _categorical, _dates, data_key: str, cfg_key: str):
    return UniversalRecommender(_df, _cfg, _numeric, _categorical, _dates)


def _clean_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in frame.columns if c.startswith("_")], errors="ignore")


def _show_ranked(frame: pd.DataFrame, target: str, score_col: str, key: str):
    """Ranked table with progress bars, bar chart and CSV download."""
    table = _clean_table(frame)
    if table.empty:
        st.info("Nothing to recommend with the current settings.")
        return
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            score_col: st.column_config.ProgressColumn(score_col, min_value=0, max_value=100, format="%.0f"),
        },
    )
    st.bar_chart(table.set_index(target)[score_col])
    st.download_button(
        "⬇ Download these recommendations",
        data=safe_download(table),
        file_name="learnwise_recommendations.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )


def render_recommendations(df, numeric_cols, categorical_cols, date_cols, data_key):
    kp = f"rec{zlib.crc32(data_key.encode()) & 0xFFFFFF:06x}"
    auto = auto_config(df, numeric_cols, categorical_cols, date_cols)
    all_cols = df.columns.tolist()
    text_like = [c for c in categorical_cols if c not in date_cols]

    st.subheader("Personalised recommendations")

    # ---- column roles (auto-detected, adjustable) ----
    with st.expander("⚙️ Detected column roles - adjust if something looks wrong", expanded=False):
        mode = st.radio(
            "Each row in my CSV is…",
            ["A student / person (profile)", "An item to recommend (course, resource, product…)"],
            index=1 if auto.catalog_mode else 0,
            horizontal=True,
            key=f"{kp}_mode",
        )
        c1, c2 = st.columns(2)
        label_options = ["(none)"] + all_cols
        label_col = c1.selectbox(
            "Name / label column",
            label_options,
            index=label_options.index(auto.label_col) if auto.label_col in label_options else 0,
            key=f"{kp}_label",
        )
        outcome_options = ["(none)"] + numeric_cols
        outcome_col = c2.selectbox(
            "Performance / rating column (quality signal)",
            outcome_options,
            index=outcome_options.index(auto.outcome_col) if auto.outcome_col in outcome_options else 0,
            key=f"{kp}_outcome",
        )
        interest_cols = c1.multiselect(
            "Interest / preference columns",
            text_like,
            default=[c for c in auto.interest_cols if c in text_like],
            key=f"{kp}_interest",
        )
        text_cols = c2.multiselect(
            "Long-text columns (descriptions)",
            text_like,
            default=[c for c in auto.text_cols if c in text_like],
            key=f"{kp}_text",
        )
        target_options = [c for c in text_like if df[c].nunique(dropna=True) > 1]
        targets = st.multiselect(
            "What should be recommended? (one or more columns)",
            target_options,
            default=[c for c in auto.targets if c in target_options],
            key=f"{kp}_targets",
            help="E.g. topic, resource format, course title.",
        )
        default_weight = 0.15 if mode.startswith("An item") else 0.4
        numeric_weight = st.slider(
            "Importance of performance / behaviour numbers vs interests & preferences",
            0.0, 1.0, float(default_weight), 0.05,
            key=f"{kp}_weight",
        )

    if not targets:
        st.warning("Choose at least one column to recommend (see column roles above).")
        return

    cfg = RecConfig(
        label_col=None if label_col == "(none)" else label_col,
        id_col=auto.id_col,
        interest_cols=interest_cols,
        text_cols=text_cols,
        targets=targets,
        outcome_col=None if outcome_col == "(none)" else outcome_col,
        catalog_mode=mode.startswith("An item"),
        numeric_weight=numeric_weight,
    )
    try:
        rec = build_engine(df, cfg, numeric_cols, categorical_cols, date_cols, data_key, cfg.key())
    except Exception as exc:  # pragma: no cover - surfaced to the user
        st.error("The recommender could not be built from these columns.")
        st.exception(exc)
        return

    st.caption(
        f"Detected: **{'catalogue of items' if cfg.catalog_mode else 'student / people records'}**  ·  "
        f"recommending **{', '.join(cfg.targets)}**  ·  "
        f"quality signal **{cfg.outcome_col or 'none'}**  ·  "
        f"{len(rec.num_feats)} numeric + {len(rec.cat_feats)} category features"
    )

    if getattr(rec, "popularity_only", False):
        st.info("This CSV has no other usable columns, so recommendations are based on popularity only. "
                "Add interest, performance or preference columns for personalised results.")

    tab_interest, tab_profile, tab_batch, tab_eval = st.tabs(
        ["🎯 By interest", "👤 For a student / profile", "📦 Whole file", "📈 Evaluation"]
    )

    # ========================= By interest =========================
    with tab_interest:
        target = st.selectbox("What should I recommend?", cfg.targets, key=f"{kp}_i_target")
        values = rec.interest_values()
        counts = dict(values)
        selected = st.multiselect(
            "Pick interests found in your data",
            [v for v, _ in values],
            format_func=lambda v: f"{v}  ({counts.get(v, 0)})",
            key=f"{kp}_i_sel",
        )
        free_text = st.text_input(
            "…or describe what you want to learn",
            placeholder="e.g. python, machine learning, beginner-friendly",
            key=f"{kp}_i_free",
        )

        filters: dict = {}
        with st.expander("Filters (optional)"):
            for col in rec.cat_feats[:6]:
                options = sorted(df[col].dropna().astype(str).unique().tolist())[:200]
                chosen = st.multiselect(col, options, key=f"{kp}_f_{col}")
                if chosen:
                    filters[col] = chosen
            if cfg.outcome_col:
                series = pd.to_numeric(df[cfg.outcome_col], errors="coerce").dropna()
                if not series.empty and series.min() < series.max():
                    lo, hi = float(series.min()), float(series.max())
                    picked = st.slider(f"Minimum {cfg.outcome_col}", lo, hi, lo, key=f"{kp}_f_out")
                    if picked > lo:
                        filters[cfg.outcome_col] = (picked, hi)

        c1, c2 = st.columns(2)
        top_n = c1.slider("How many recommendations", 3, 20, 8, key=f"{kp}_i_n")
        diversity = 0.0
        if cfg.catalog_mode:
            diversity = c2.slider("Variety (0 = most similar, 1 = most varied)", 0.0, 1.0, 0.2, 0.05,
                                  key=f"{kp}_i_div")

        result = rec.recommend_by_interest(selected, free_text, target, n=top_n, filters=filters,
                                           diversity=diversity)
        if result.attrs.get("note"):
            st.info(result.attrs["note"])

        if len(result):
            top_cards = st.columns(min(3, len(result)))
            for container, (_, row) in zip(top_cards, result.head(3).iterrows()):
                with container:
                    metric_card(f"#{int(row['Rank'])} pick", str(row[target]))

            shown = result.copy()
            if cfg.catalog_mode and shown["_row"].notna().all():
                details = df.loc[shown["_row"].astype(int).tolist()].reset_index(drop=True)
                drop_cols = [c for c in details.columns if c == target or c in cfg.text_cols or c == auto.id_col]
                details = details.drop(columns=drop_cols, errors="ignore")
                shown = pd.concat([shown.reset_index(drop=True), details], axis=1)
            st.markdown("### Your recommendations")
            _show_ranked(shown, target, "Match score", "interest")

        if not cfg.catalog_mode and cfg.interest_cols and df[target].nunique() <= 30:
            with st.expander("Interest × recommendation heatmap"):
                heat_col = st.selectbox("Interest column", cfg.interest_cols, key=f"{kp}_heat_col")
                if cfg.outcome_col:
                    measure = st.radio("Show", ["Number of students", f"Average {cfg.outcome_col}"],
                                       horizontal=True, key=f"{kp}_heat_m")
                else:
                    measure = "Number of students"
                if measure == "Number of students":
                    grid = pd.crosstab(df[heat_col], df[target])
                else:
                    grid = df.pivot_table(index=heat_col, columns=target, values=cfg.outcome_col,
                                          aggfunc="mean").round(1)
                try:
                    st.dataframe(grid.style.background_gradient(cmap="Blues", axis=None),
                                 use_container_width=True)
                except Exception:
                    st.dataframe(grid, use_container_width=True)

    # ========================= For a student / profile =========================
    with tab_profile:
        k_neighbors = st.slider("How many similar records to learn from", 5, 100, 25, key=f"{kp}_p_k")
        n_recs = st.slider("Recommendations per target", 1, 10, 5, key=f"{kp}_p_n")

        if cfg.catalog_mode:
            st.write("Pick items you liked and get **more like these**.")
            limit = min(len(df), 5000)
            liked = st.multiselect(
                "Items I liked",
                list(range(limit)),
                format_func=lambda i: rec.label_of(i),
                key=f"{kp}_p_liked",
            )
            if liked:
                for tgt in cfg.targets:
                    out = rec.recommend_from_rows(liked, tgt, n=n_recs, k=k_neighbors)
                    st.markdown(f"### More like your picks · {tgt}")
                    _show_ranked(out, tgt, "Match %", f"liked_{tgt}")
            else:
                st.info("Choose one or more items above.")
        else:
            source = st.radio("Profile source", ["Existing record from the file", "Build a custom profile"],
                              horizontal=True, key=f"{kp}_p_src")
            record: dict = {}
            outs: dict = {}
            sims = None
            if source.startswith("Existing"):
                limit = min(len(df), 5000)
                idx = st.selectbox("Choose a record", list(range(limit)),
                                   format_func=lambda i: rec.label_of(i), key=f"{kp}_p_row")
                exclude_current = st.checkbox("Suggest something new (skip what this record already has)",
                                              value=False, key=f"{kp}_p_ex")
                record = df.iloc[idx].to_dict()
                with st.expander("Selected record", expanded=False):
                    st.dataframe(df.iloc[[idx]], use_container_width=True, hide_index=True)
                for tgt in cfg.targets:
                    outs[tgt] = rec.recommend_for_record(idx, tgt, n=n_recs, k=k_neighbors,
                                                         exclude_current=exclude_current)
                sims = outs[cfg.targets[0]].attrs.get("sims")
            else:
                st.write("Enter a learner profile (leave categories on *(any)* to ignore them).")
                profile: dict = {}
                cols = st.columns(3)
                for j, col in enumerate(rec.num_feats):
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    lo, hi, med = float(series.min()), float(series.max()), float(series.median())
                    with cols[j % 3]:
                        profile[col] = st.slider(col, lo, hi, med, key=f"{kp}_pf_{col}")
                for j, col in enumerate(rec.cat_feats):
                    options = ["(any)"] + sorted(df[col].dropna().astype(str).unique().tolist())
                    with cols[j % 3]:
                        profile[col] = st.selectbox(col, options, key=f"{kp}_pf_{col}")
                free = ""
                record = profile
                for tgt in cfg.targets:
                    outs[tgt] = rec.recommend_for_profile(profile, tgt, n=n_recs, k=k_neighbors, free_text=free)
                sims = outs[cfg.targets[0]].attrs.get("sims")

            panels = st.columns(len(cfg.targets))
            for container, tgt in zip(panels, cfg.targets):
                with container:
                    st.markdown(f"#### Recommended {tgt}")
                    out = outs[tgt]
                    if out.attrs.get("note"):
                        st.caption(out.attrs["note"])
                    _show_ranked(out, tgt, "Match %", f"profile_{tgt}")

            tips = rec.guidance(record)
            if tips:
                st.markdown("#### Study guidance")
                for tip in tips:
                    st.markdown(f"- {tip}")
            if sims is not None:
                with st.expander("Most similar records used as evidence"):
                    st.dataframe(rec.similar_rows(sims, 8), use_container_width=True, hide_index=True)

    # ========================= Whole file =========================
    with tab_batch:
        st.write(
            "Generate recommendations for **every row** of the loaded CSV and download the result."
        )
        c1, c2 = st.columns(2)
        batch_targets = c1.multiselect("Targets", cfg.targets, default=cfg.targets, key=f"{kp}_b_targets")
        n_choices = c2.slider("Choices per row", 1, 5, 3, key=f"{kp}_b_n")
        skip_current = st.checkbox("Skip what each row already has (suggest something new)", value=False,
                                   key=f"{kp}_b_skip")
        if len(df) > 20000:
            st.caption("Large file: recommendations are generated for the first 20,000 rows.")
        if st.button("Generate for all rows", type="primary", key=f"{kp}_b_go") and batch_targets:
            bar = st.progress(0.0)
            st.session_state[f"{kp}_batch"] = rec.recommend_all(
                batch_targets, n=n_choices, exclude_current=skip_current,
                progress=lambda value: bar.progress(float(value)),
            )
            bar.empty()
        batch = st.session_state.get(f"{kp}_batch")
        if batch is not None:
            st.success(f"Created recommendations for {len(batch):,} rows.")
            st.dataframe(batch, use_container_width=True, hide_index=True, height=420)
            first_choices = [c for c in batch.columns if c.endswith("1st choice")]
            if first_choices:
                pick = st.selectbox("Distribution of the 1st recommendation", first_choices, key=f"{kp}_b_view")
                st.bar_chart(batch[pick].replace("", np.nan).dropna().value_counts())
            st.download_button(
                "⬇ Download recommendations for all rows",
                data=safe_download(batch),
                file_name="learnwise_all_recommendations.csv",
                mime="text/csv",
                key=f"{kp}_b_dl",
            )

    # ========================= Evaluation =========================
    with tab_eval:
        st.write(
            "Hold out 20% of the rows, hide their answer, and check whether the recommender ranks the "
            "true value highly - compared with simply suggesting the most popular value."
        )
        eval_target = st.selectbox("Target to evaluate", cfg.targets, key=f"{kp}_e_target")
        top_n_eval = st.slider("Hit@N - is the true value within the top N?", 1, 5, 3, key=f"{kp}_e_n")
        report = rec.evaluate(eval_target, k=25, top_n=top_n_eval)
        if report is None:
            st.info(
                "Offline evaluation needs a target whose values repeat across many rows (at least 30 rows, "
                "and each value used about twice or more). A catalogue of unique items cannot be evaluated this way."
            )
        else:
            model, base = report["model"], report["baseline"]
            a, b, c, d = st.columns(4)
            with a:
                metric_card("Hit@1", f"{model['hit1']:.0%}")
            with b:
                metric_card(f"Hit@{top_n_eval}", f"{model['hitn']:.0%}")
            with c:
                metric_card("MRR", f"{model['mrr']:.2f}")
            with d:
                metric_card("Popularity baseline Hit@1", f"{base['hit1']:.0%}")

            compare = pd.DataFrame(
                {
                    "Recommender": [model["hit1"], model["hitn"], model["mrr"]],
                    "Popularity baseline": [base["hit1"], base["hitn"], base["mrr"]],
                },
                index=["Hit@1", f"Hit@{top_n_eval}", "MRR"],
            )
            st.bar_chart(compare)
            lift = model["hit1"] - base["hit1"]
            if lift > 0.05:
                st.success(f"The recommender beats the popularity baseline by {lift * 100:.0f} percentage points "
                           "on held-out rows.")
            else:
                st.warning("The recommender is not clearly better than suggesting the most popular value. "
                           "The target may be random, or the columns may not carry enough signal.")
            st.caption(
                f"{report['n_train']} training rows · {report['n_test']} test rows · "
                f"{report['classes']} possible values · random guessing Hit@1 ≈ {report['random_hit1']:.0%}"
            )
            left, right = st.columns(2)
            with left:
                st.markdown("#### Confusion matrix (top-1)")
                try:
                    st.dataframe(report["confusion"].style.background_gradient(cmap="Blues", axis=None),
                                 use_container_width=True)
                except Exception:
                    st.dataframe(report["confusion"], use_container_width=True)
            with right:
                st.markdown("#### Recall by value")
                st.dataframe(report["per_class"], use_container_width=True)

    st.caption(
        "Recommendations are suggestions, not decisions - teachers and students should be able to override them. "
        "Keep student identifiers private and avoid using these outputs to grade or rank people."
    )


# --------------------------------- Recommendations -------------------------

if page == "Recommendations":
    data_key = f"{uploaded.name}|{len(uploaded.getvalue())}|{delimiter_choice}|{header_row}|{skip_bad_lines}"
    render_recommendations(df, numeric_cols, categorical_cols, date_cols, data_key)


# --------------------------------- Overview --------------------------------

elif page == "Overview":
    st.subheader("Dataset overview")

    cells = [
        ("Total rows", f"{len(df):,}"),
        ("Total columns", f"{len(all_columns):,}"),
        ("Numeric columns", f"{len(numeric_cols):,}"),
        ("Missing cells", f"{int(df.isna().sum().sum()):,}"),
    ]
    metric_cols = st.columns(4)
    for container, (label, value) in zip(metric_cols, cells):
        with container:
            metric_card(label, value)

    st.markdown("### Data preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Missing values by column")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].head(20)
        if not missing.empty:
            st.bar_chart(missing)
        else:
            st.success("No missing values detected.")

    with right:
        st.markdown("#### Explore a numeric distribution")
        if numeric_cols:
            selected = st.selectbox(
                "Choose a numeric column",
                numeric_cols,
                key="overview_numeric",
            )
            values = pd.to_numeric(df[selected], errors="coerce").dropna()
            if not values.empty:
                bin_count = min(20, max(5, int(values.nunique())))

                histogram = pd.cut(
                    values,
                    bins=bin_count,
                    duplicates="drop"
                ).value_counts().sort_index()

                histogram.index = histogram.index.astype(str)

                st.bar_chart(histogram)
            else:
                st.info("This column has no usable numeric values.")
        else:
            st.info("No numeric columns were detected.")

    if categorical_cols:
        st.markdown("#### Category distribution")
        selected_cat = st.selectbox(
            "Choose a text/category column",
            categorical_cols,
            key="overview_category",
        )
        st.bar_chart(category_counts(df[selected_cat]))


# --------------------------------- Explorer --------------------------------

elif page == "Data explorer":
    st.subheader("Search, filter and export records")

    result = df.copy()
    search_text = st.text_input(
        "Search across all columns",
        placeholder="Type a name, ID, category, number or any value...",
    )
    if search_text.strip():
        search_mask = result.astype("string").apply(
            lambda column: column.str.contains(
                search_text.strip(), case=False, na=False, regex=False
            )
        ).any(axis=1)
        result = result[search_mask]

    with st.expander("Filter records", expanded=True):
        selected_filters = st.multiselect(
            "Select columns to filter",
            all_columns,
            help="You can apply multiple filters together.",
        )

        for col in selected_filters:
            if col in numeric_cols:
                values = pd.to_numeric(df[col], errors="coerce").dropna()
                if values.empty:
                    st.caption(f"{col}: no numeric values to filter.")
                    continue
                minimum, maximum = float(values.min()), float(values.max())
                if minimum < maximum:
                    selected_range = st.slider(
                        f"{col} range",
                        min_value=minimum,
                        max_value=maximum,
                        value=(minimum, maximum),
                        key=f"filter_numeric_{col}",
                    )
                    result = result[
                        pd.to_numeric(result[col], errors="coerce").between(
                            selected_range[0], selected_range[1]
                        )
                    ]

            elif col in date_cols:
                values = pd.to_datetime(df[col], errors="coerce").dropna()
                if values.empty:
                    continue
                selected_dates = st.date_input(
                    f"{col} date range",
                    value=(values.min().date(), values.max().date()),
                    key=f"filter_date_{col}",
                )
                if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                    dates = pd.to_datetime(result[col], errors="coerce")
                    start = pd.Timestamp(selected_dates[0])
                    end = pd.Timestamp(selected_dates[1]) + pd.Timedelta(days=1)
                    result = result[dates.ge(start) & dates.lt(end)]

            else:
                options = (
                    df[col].dropna().astype(str).value_counts().head(500).index.tolist()
                )
                chosen_values = st.multiselect(
                    f"{col} values",
                    options,
                    key=f"filter_category_{col}",
                )
                if chosen_values:
                    result = result[result[col].astype(str).isin(chosen_values)]

    st.write(f"**{len(result):,} matching rows** out of {len(df):,}")
    st.dataframe(result, use_container_width=True, hide_index=True, height=520)

    st.download_button(
        "⬇ Download filtered CSV",
        data=safe_download(result),
        file_name="learnwise_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------------- Visual analytics ------------------------

elif page == "Visual analytics":
    st.subheader("Create charts from your dataset")
    chart_type = st.selectbox(
        "Chart type",
        ["Bar chart", "Line chart", "Histogram", "Scatter plot", "Category frequency"],
    )

    if chart_type == "Histogram":
        if not numeric_cols:
            st.warning("A histogram needs at least one numeric column.")
        else:
            column = st.selectbox("Numeric column", numeric_cols)
            bins = st.slider("Number of bins", 5, 60, 20)
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                counts = pd.cut(values, bins=bins, duplicates="drop").value_counts().sort_index()
                counts.index = counts.index.astype(str)
                st.bar_chart(counts)
            else:
                st.info("No numeric values are available in this column.")

    elif chart_type in ("Bar chart", "Line chart"):
        if not numeric_cols:
            st.warning("Bar and line charts need at least one numeric column.")
        else:
            x_column = st.selectbox("X axis (category, date or other field)", all_columns)
            y_column = st.selectbox("Y axis (numeric field)", numeric_cols)
            aggregation = st.selectbox(
                "Aggregation",
                ["Mean", "Sum", "Count", "Minimum", "Maximum"],
            )

            chart_data = df[[x_column, y_column]].copy()
            chart_data[y_column] = pd.to_numeric(chart_data[y_column], errors="coerce")
            chart_data = chart_data.dropna(subset=[x_column, y_column])

            if not chart_data.empty:
                if chart_data[x_column].nunique() > 100:
                    st.warning("The X column has many unique values. Showing the 100 most common.")
                    top_values = chart_data[x_column].value_counts().head(100).index
                    chart_data = chart_data[chart_data[x_column].isin(top_values)]

                grouped = chart_data.groupby(x_column, dropna=False)[y_column]
                if aggregation == "Mean":
                    plotted = grouped.mean()
                elif aggregation == "Sum":
                    plotted = grouped.sum()
                elif aggregation == "Count":
                    plotted = grouped.count()
                elif aggregation == "Minimum":
                    plotted = grouped.min()
                else:
                    plotted = grouped.max()

                plotted = plotted.sort_values(ascending=False) if chart_type == "Bar chart" else plotted.sort_index()
                st.caption(f"{aggregation} of {y_column} grouped by {x_column}")
                if chart_type == "Bar chart":
                    st.bar_chart(plotted)
                else:
                    st.line_chart(plotted)
            else:
                st.info("No valid records are available for the selected fields.")

    elif chart_type == "Scatter plot":
        if len(numeric_cols) < 2:
            st.warning("A scatter plot needs at least two numeric columns.")
        else:
            left, right = st.columns(2)
            x_column = left.selectbox("X axis", numeric_cols, key="scatter_x")
            y_options = [column for column in numeric_cols if column != x_column]
            y_column = right.selectbox("Y axis", y_options, key="scatter_y")
            points = df[[x_column, y_column]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(points) > 5000:
                points = points.sample(5000, random_state=42)
                st.caption("Showing a random sample of 5,000 rows for performance.")
            if not points.empty:
                st.scatter_chart(points, x=x_column, y=y_column, use_container_width=True)
            else:
                st.info("No valid numeric pairs were found.")

    else:
        if not categorical_cols:
            st.warning("No text/category columns were detected.")
        else:
            column = st.selectbox("Category column", categorical_cols)
            limit = st.slider("Number of categories", 5, 50, 15)
            st.bar_chart(category_counts(df[column], limit=limit))


# --------------------------------- Column insights -------------------------

elif page == "Column insights":
    st.subheader("Understand an individual column")
    selected_column = st.selectbox("Choose a column", all_columns)
    series = df[selected_column]

    a, b, c, d = st.columns(4)
    a.metric("Non-empty", f"{series.notna().sum():,}")
    b.metric("Missing", f"{series.isna().sum():,}")
    c.metric("Unique values", f"{series.nunique(dropna=True):,}")
    d.metric("Data type", str(series.dtype))

    if selected_column in numeric_cols:
        values = pd.to_numeric(series, errors="coerce")
        st.markdown("#### Statistical summary")
        st.dataframe(values.describe().to_frame("Value"), use_container_width=True)
        valid = values.dropna()
        if not valid.empty:
            bins = st.slider("Histogram bins", 5, 60, 20)
            counts = pd.cut(valid, bins=bins, duplicates="drop").value_counts().sort_index()
            counts.index = counts.index.astype(str)
            st.bar_chart(counts)

    elif selected_column in date_cols:
        values = pd.to_datetime(series, errors="coerce").dropna()
        if not values.empty:
            st.write(f"**Earliest:** {values.min()}  |  **Latest:** {values.max()}")
            monthly = values.dt.to_period("M").astype(str).value_counts().sort_index()
            st.markdown("#### Records by month")
            st.bar_chart(monthly)
        else:
            st.info("No valid dates were found.")

    else:
        counts = category_counts(series, limit=50)
        st.markdown("#### Most frequent values")
        st.bar_chart(counts)
        st.dataframe(counts.rename("Count").to_frame(), use_container_width=True)


# --------------------------------- Data quality ----------------------------

elif page == "Data quality":
    st.subheader("Data quality report")

    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    complete_count = total_cells - missing_count

    a, b, c, d = st.columns(4)
    a.metric("Missing cells", f"{missing_count:,}")
    b.metric("Duplicate rows", f"{duplicate_count:,}")
    c.metric("Complete cells", f"{complete_count:,}")
    d.metric("Completeness", f"{(complete_count / total_cells * 100):.1f}%" if total_cells else "N/A")

    report = pd.DataFrame(
        {
            "Column": all_columns,
            "Detected type": [str(df[column].dtype) for column in all_columns],
            "Non-empty": [int(df[column].notna().sum()) for column in all_columns],
            "Missing": [int(df[column].isna().sum()) for column in all_columns],
            "Missing (%)": [
                round(float(df[column].isna().mean() * 100), 2)
                for column in all_columns
            ],
            "Unique": [int(df[column].nunique(dropna=True)) for column in all_columns],
        }
    )

    st.markdown("#### Column quality")
    st.dataframe(report, use_container_width=True, hide_index=True)

    missing_percent = (df.isna().mean() * 100).sort_values(ascending=False)
    missing_percent = missing_percent[missing_percent > 0]
    if not missing_percent.empty:
        st.markdown("#### Missing data percentage")
        st.bar_chart(missing_percent)
    else:
        st.success("No missing values detected.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Duplicate row handling")
        st.write(f"Found **{duplicate_count:,}** fully duplicated rows.")
        if duplicate_count:
            if st.button("Create duplicate-free export", use_container_width=True):
                deduplicated = df.drop_duplicates()
                st.session_state["learnwise_deduplicated"] = deduplicated
                st.success(f"Prepared {len(deduplicated):,} unique rows.")
        if "learnwise_deduplicated" in st.session_state:
            st.download_button(
                "⬇ Download duplicate-free CSV",
                data=safe_download(st.session_state["learnwise_deduplicated"]),
                file_name="learnwise_deduplicated.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with right:
        st.markdown("#### Export current dataset")
        st.download_button(
            "⬇ Download loaded dataset",
            data=safe_download(df),
            file_name="learnwise_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --------------------------------- Footer ----------------------------------

st.divider()
st.caption(
    f"LearnWise v{APP_VERSION} · Personalised recommendations · Flexible CSV import · "
    "Exploration · Visualization · Data quality"
)
