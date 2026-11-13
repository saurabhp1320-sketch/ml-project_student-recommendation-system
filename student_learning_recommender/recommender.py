"""
LearnWise universal recommendation engine
=========================================

Works with ANY CSV. It auto-detects which columns describe interests, free text,
performance/ratings, identifiers and "things to recommend", and offers:

1. recommend_by_interest()   - "I like X, Y, Z" -> best matching items / topics / formats
                               (TF-IDF word + character n-grams, rating/performance aware,
                               optional diversity re-ranking, explainable "why").
2. recommend_for_record() /
   recommend_for_profile()   - "students like me" -> k-nearest-neighbour recommendation built
                               from behaviour (numeric), interests and preferences (categorical).
3. recommend_all()           - batch recommendations for every row of the CSV.
4. evaluate()                - offline hold-out evaluation (Hit@1, Hit@N, MRR) vs a popularity baseline.
5. guidance()                - plain-language study guidance from performance / behaviour columns.

Only numpy, pandas, scipy and scikit-learn are required.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

_SPLIT = re.compile(r"[;,|/]+")
_WORD = re.compile(r"\w+", re.UNICODE)

ID_TOKENS = {"id", "uuid", "guid", "sku", "code", "index", "idx", "roll", "no", "num", "number", "key"}
INTEREST_HINTS = (
    "interest", "hobby", "hobbies", "genre", "categor", "skill", "tag", "keyword", "prefer",
    "domain", "field", "area", "major", "department", "theme", "goal", "stream", "specializ",
)
STRONG_TARGET_HINTS = ("recommend", "best", "next", "suggest", "target", "label")
WEAK_TARGET_HINTS = ("topic", "resource", "course", "subject", "material", "content", "module", "lesson")
LABEL_HINTS = ("title", "name", "course", "resource", "item", "product", "movie", "book", "lesson",
               "module", "material", "song", "video", "article")
CATALOG_LABEL_HINTS = tuple(h for h in LABEL_HINTS if h != "name")
OUTCOME_HINTS = ("score", "rating", "grade", "mark", "star", "gpa", "perform", "quiz", "accuracy",
                 "result", "rank", "review", "percent")
STOP_WORDS = frozenset(
    "a an and are as at be but by for from has have i im in into is it its me my of on or our so that the "
    "their them these this to was we what which with would you your want wants need needs like likes learn "
    "learning about interested interest looking find show give recommend suggest please course courses topic "
    "topics study studying more some any good best".split()
) - {"learning"}
BEHAVIOUR_HINTS = ("attend", "study", "hour", "complet", "engage", "practice", "login", "streak",
                   "submi", "particip", "time_spent", "minutes")


# --------------------------------------------------------------------------- #
# Column detection
# --------------------------------------------------------------------------- #
def _name_tokens(name: Any) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _has_hint(name: Any, hints: tuple[str, ...]) -> bool:
    low = str(name).lower()
    return any(h in low for h in hints)


def _is_multivalued(series: pd.Series) -> bool:
    values = series.dropna().astype(str)
    if values.empty:
        return False
    return bool(values.str.contains(r"[;,|]", regex=True).mean() > 0.2)


@dataclass
class RecConfig:
    label_col: Optional[str] = None          # human readable row name (title / student name / id)
    id_col: Optional[str] = None
    interest_cols: list = field(default_factory=list)
    text_cols: list = field(default_factory=list)
    targets: list = field(default_factory=list)   # columns whose values we recommend
    outcome_col: Optional[str] = None        # rating / score / grade used as a quality signal
    catalog_mode: bool = False               # True: rows are items; False: rows are people/records
    numeric_weight: float = 0.4

    def key(self) -> str:
        return repr(asdict(self))


def auto_config(df: pd.DataFrame, numeric_cols, categorical_cols, date_cols=()) -> RecConfig:
    """Infer sensible roles for every column of an arbitrary CSV."""
    n = len(df)
    numeric_cols, date_cols = list(numeric_cols), set(date_cols)
    id_col, label_candidates, text_cols, cat_cols = None, [], [], []

    for col in df.columns:
        series = df[col]
        tokens = set(_name_tokens(col))
        nunique = series.nunique(dropna=True)
        ratio = nunique / max(n, 1)
        if col in date_cols:
            continue
        if col in numeric_cols:
            if tokens & ID_TOKENS and ratio > 0.9 and id_col is None:
                id_col = col
            continue
        values = series.dropna().astype(str)
        avg_words = float(values.str.split().str.len().mean()) if len(values) else 0.0
        if tokens & ID_TOKENS and ratio > 0.5:
            id_col = id_col or col
        elif avg_words >= 5 and ratio > 0.5:
            text_cols.append(col)
        elif ratio > 0.9 and n >= 10:
            label_candidates.append(col)
        elif 1 < nunique:
            cat_cols.append(col)

    label_col = None
    for col in label_candidates:
        if _has_hint(col, LABEL_HINTS):
            label_col = col
            break
    if label_col is None and label_candidates:
        label_col = label_candidates[0]
    display = label_col or id_col

    catalog_mode = bool(
        label_col and (text_cols or _has_hint(label_col, CATALOG_LABEL_HINTS))
    )

    # outcome (rating / score)
    outcome = None
    scored = [c for c in numeric_cols if c != id_col and _has_hint(c, OUTCOME_HINTS)]
    if scored:
        outcome = scored[0]

    # targets
    if catalog_mode:
        targets = [label_col]
    else:
        strong = [c for c in cat_cols if _has_hint(c, STRONG_TARGET_HINTS) and "prefer" not in str(c).lower()]
        weak = [c for c in cat_cols if _has_hint(c, WEAK_TARGET_HINTS) and c not in strong
                and "prefer" not in str(c).lower()]
        targets = (strong + weak)[:3]
        if not targets and cat_cols:
            targets = [min(cat_cols, key=lambda c: df[c].nunique())]

    interest = [c for c in cat_cols if c not in targets and _has_hint(c, INTEREST_HINTS)]
    if not interest:
        interest = [c for c in cat_cols if c not in targets]

    return RecConfig(
        label_col=display, id_col=id_col, interest_cols=interest, text_cols=text_cols,
        targets=targets, outcome_col=outcome, catalog_mode=catalog_mode,
        numeric_weight=0.15 if catalog_mode else 0.4,
    )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class UniversalRecommender:
    def __init__(self, df: pd.DataFrame, cfg: RecConfig, numeric_cols, categorical_cols, date_cols=()):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.n = len(self.df)
        self.date_cols = set(date_cols)
        numeric_cols, categorical_cols = list(numeric_cols), list(categorical_cols)
        skip = {c for c in [cfg.id_col] if c} | set(cfg.text_cols) | self.date_cols
        targets = set(cfg.targets)

        self.num_feats = [
            c for c in numeric_cols
            if c not in skip and c not in targets and self.df[c].notna().sum() > 1 and self.df[c].nunique() > 1
        ]
        self.cat_feats = [
            c for c in categorical_cols
            if c not in skip and c not in targets and c != cfg.label_col
            and 1 < self.df[c].nunique(dropna=True) <= 60
        ]
        # columns that contribute to the text documents
        self.doc_cols = [
            c for c in categorical_cols
            if c not in skip or c == cfg.label_col
        ]
        self.multi = {
            c: _is_multivalued(self.df[c])
            for c in set(self.doc_cols) | set(self.cat_feats) if c not in cfg.text_cols
        }

        self._prepare_outcome()
        self._build_documents()
        self._build_feature_blocks()
        self._assemble()
        self._codes_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    # ------------------------------------------------------------------ prep
    def _prepare_outcome(self):
        oc = self.cfg.outcome_col
        if oc and oc in self.df.columns:
            raw = pd.to_numeric(self.df[oc], errors="coerce")
            self.o_raw = raw.to_numpy(float)
            self.o_pct = raw.rank(pct=True).fillna(0.5).to_numpy(float)
        else:
            self.o_raw = None
            self.o_pct = np.full(self.n, 0.5)

    def _cell_tokens(self, value, multi: bool = False, phrase: bool = True, drop_stop: bool = False) -> list[str]:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return []
        text = str(value).lower().strip()
        parts = [p.strip() for p in _SPLIT.split(text)] if multi else [text]
        out: list[str] = []
        for part in parts:
            if not part:
                continue
            words = _WORD.findall(part)
            if drop_stop:
                words = [w for w in words if w not in STOP_WORDS]
            out.extend(words)
            if phrase and len(words) > 1 and len(part) <= 48:
                out.append("_".join(words))
        return out

    def _cat_tokens(self, value, col: str) -> list[str]:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return []
        text = str(value).strip().lower()
        if not text or text == "nan":
            return []
        if self.multi.get(col):
            return [p.strip() for p in _SPLIT.split(text) if p.strip()]
        return [text]

    def _build_documents(self):
        cfg = self.cfg
        word_docs: list[list[str]] = [[] for _ in range(self.n)]
        short_docs: list[list[str]] = [[] for _ in range(self.n)]
        for col in self.doc_cols:
            weight = 2 if col in cfg.interest_cols else 1
            multi = self.multi.get(col, False)
            tokens = self.df[col].map(lambda v, m=multi: self._cell_tokens(v, m)).tolist()
            for i, toks in enumerate(tokens):
                word_docs[i].extend(toks * weight)
                short_docs[i].extend(toks * weight)
        for col in cfg.text_cols:
            tokens = self.df[col].map(lambda v: self._cell_tokens(v, False, phrase=False, drop_stop=True)).tolist()
            for i, toks in enumerate(tokens):
                word_docs[i].extend(toks)

        self.tf_word = self.tf_char = None
        self.D_word = self.D_char = None
        self._word_names = None
        try:
            self.tf_word = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
            self.D_word = self.tf_word.fit_transform([" ".join(d) for d in word_docs]).tocsr()
            self._word_names = self.tf_word.get_feature_names_out()
        except ValueError:
            self.tf_word = self.D_word = None
        try:
            self.tf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                                           sublinear_tf=True, max_features=60000)
            self.D_char = self.tf_char.fit_transform([" ".join(d) for d in short_docs]).tocsr()
        except ValueError:
            self.tf_char = self.D_char = None

    @staticmethod
    def _num_embed(z: np.ndarray) -> np.ndarray:
        """Embed z-scores so that cosine similarity reflects closeness, not just direction."""
        d = max(z.shape[1], 1)
        emb = np.hstack([z / (2.0 * np.sqrt(d)), np.ones((z.shape[0], 1))])
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def _build_feature_blocks(self):
        cfg = self.cfg
        self.Xnum = None
        if self.num_feats:
            x = self.df[self.num_feats].apply(pd.to_numeric, errors="coerce")
            self.num_median = x.median().fillna(0.0)
            x = x.fillna(self.num_median)
            self.num_mean = x.mean()
            self.num_std = x.std(ddof=0).replace(0, 1).fillna(1)
            z = ((x - self.num_mean) / self.num_std).clip(-3, 3).to_numpy(float)
            self.Xnum = sp.csr_matrix(self._num_embed(z))

        self.cat_vecs: list[tuple[str, CountVectorizer, float]] = []
        blocks = []
        for col in self.cat_feats:
            weight = 1.5 if col in cfg.interest_cols else 1.0
            cv = CountVectorizer(analyzer=lambda s, c=col: self._cat_tokens(s, c), binary=True)
            docs = self.df[col].astype("object").where(self.df[col].notna(), "").astype(str)
            try:
                block = cv.fit_transform(docs)
            except ValueError:
                continue
            self.cat_vecs.append((col, cv, weight))
            blocks.append(block * weight)
        self.Xcat = normalize(sp.hstack(blocks, format="csr")) if blocks else None

    def _assemble(self):
        cfg, wn = self.cfg, float(self.cfg.numeric_weight)
        blocks: dict[str, sp.spmatrix] = {}
        if self.Xnum is not None:
            blocks["num"] = self.Xnum
        if self.Xcat is not None:
            blocks["cat"] = self.Xcat
        if cfg.catalog_mode and self.D_word is not None:
            blocks["text"] = self.D_word
        self.popularity_only = not blocks
        if self.popularity_only:
            # No feature columns at all: every row looks identical, so recommendations fall back to popularity.
            blocks["cat"] = sp.csr_matrix(np.ones((self.n, 1)))
        base = {
            "num": wn,
            "cat": (1 - wn) * (0.3 if cfg.catalog_mode else 1.0),
            "text": (1 - wn) * 0.7 if cfg.catalog_mode else 0.0,
        }
        present = {k: base[k] for k in blocks if base[k] > 0} or {k: 1.0 for k in blocks}
        total = sum(present.values())
        self.block_order = list(present)
        self.block_w = {k: v / total for k, v in present.items()}
        self.M = sp.hstack(
            [blocks[k] * np.sqrt(self.block_w[k]) for k in self.block_order], format="csr"
        ).astype(np.float32)

    # ----------------------------------------------------------------- utils
    def _codes(self, target: str):
        if target not in self._codes_cache:
            cat = pd.Categorical(self.df[target].astype("string").str.strip())
            self._codes_cache[target] = (np.asarray(cat.codes, dtype=int), [str(c) for c in cat.categories])
        return self._codes_cache[target]

    def label_of(self, i: int) -> str:
        col = self.cfg.label_col
        return str(self.df.at[i, col]) if col else f"Row {i + 1}"

    def interest_values(self, top: int = 300) -> list[tuple[str, int]]:
        cols = self.cfg.interest_cols or self.cat_feats
        counts: dict[str, int] = {}
        for col in cols:
            multi = self.multi.get(col, False)
            for v in self.df[col].dropna().astype(str):
                parts = [p.strip() for p in _SPLIT.split(v)] if multi else [v.strip()]
                for p in parts:
                    if p:
                        counts[p] = counts.get(p, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]

    def _filter_mask(self, filters: Optional[dict]) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        for col, val in (filters or {}).items():
            if col not in self.df.columns:
                continue
            s = self.df[col]
            if isinstance(val, tuple) and len(val) == 2 and pd.api.types.is_numeric_dtype(s):
                mask &= pd.to_numeric(s, errors="coerce").between(val[0], val[1]).to_numpy()
            elif val:
                mask &= s.astype(str).isin([str(v) for v in val]).to_numpy()
        return mask

    # ------------------------------------------------- 1. interest recommender
    def make_query(self, selected, free_text: str = "") -> list[str]:
        toks: list[str] = []
        for v in selected or []:
            toks += self._cell_tokens(v, False) * 2
        for phrase in re.split(r"[;,\n]+", free_text or ""):
            toks += self._cell_tokens(phrase, False, drop_stop=True)
        return toks

    def _text_similarity(self, toks: list[str]) -> np.ndarray:
        sim = np.zeros(self.n)
        if not toks:
            return sim
        doc = " ".join(toks)
        parts = []
        if self.tf_word is not None:
            qw = self.tf_word.transform([doc])
            parts.append((0.75, (self.D_word @ qw.T).toarray().ravel()))
        if self.tf_char is not None:
            qc = self.tf_char.transform([doc])
            parts.append((0.25, (self.D_char @ qc.T).toarray().ravel()))
        if not parts:
            return sim
        wsum = sum(w for w, _ in parts)
        sim = sum(w * s for w, s in parts) / wsum
        if sim.max() > 0:
            sim[sim < max(0.02, 0.1 * sim.max())] = 0.0
        return sim

    def _why(self, toks: list[str], row: int) -> str:
        if self.tf_word is None or not toks:
            return ""
        qv = self.tf_word.transform([" ".join(toks)])
        prod = qv.multiply(self.D_word[row]).tocsr()
        if prod.nnz == 0:
            return ""
        order = np.argsort(-prod.data)[:3]
        return ", ".join(str(self._word_names[prod.indices[i]]).replace("_", " ") for i in order)

    def recommend_by_interest(self, selected, free_text, target, n=10, filters=None, diversity=0.0):
        toks = self.make_query(selected, free_text)
        mask = self._filter_mask(filters)
        codes, cats = self._codes(target)
        valid = (codes >= 0) & mask
        sims = self._text_similarity(toks) * valid
        note = None
        if sims.max() <= 0:
            note = ("No rows matched those interests (after filters) - showing the best overall "
                    "options instead." if toks else "No interests given - showing the best overall options.")
            sims = valid.astype(float) * 1e-3
            toks = []

        C = len(cats)
        vc = codes[valid]
        n_g = np.bincount(vc, minlength=C).astype(float)
        s_g = np.bincount(vc, weights=sims[valid], minlength=C)
        so_g = np.bincount(vc, weights=(sims * self.o_pct)[valid], minlength=C)
        matches = np.bincount(codes[(sims > 0) & valid], minlength=C)
        rel = np.divide(s_g, np.sqrt(np.maximum(n_g, 1)))
        m = np.minimum(3.0, np.maximum(n_g - 1, 0))
        qual = np.divide(so_g + m * 0.5, s_g + m, out=np.full(C, 0.5), where=(s_g + m) > 0)
        rel_n = rel / rel.max() if rel.max() > 0 else rel
        final = (0.7 * rel_n + 0.3 * qual) if self.o_raw is not None else rel_n
        final = np.where(s_g > 0, final, -1)

        order = np.argsort(-final)
        order = [c for c in order if final[c] >= 0]
        # best (most similar) row per class, used for "why" and for item details
        best: dict[int, int] = {}
        for i in np.argsort(-sims):
            if not valid[i]:
                continue
            best.setdefault(int(codes[i]), int(i))
            if len(best) == C:
                break

        if diversity > 0 and n_g.max() == 1 and len(order) > 1:
            order = self._mmr(order, final, best, n, 1 - diversity)

        rows = []
        for rank, c in enumerate(order[:n], 1):
            b = best.get(int(c))
            rows.append({
                "Rank": rank,
                target: cats[c],
                "Match score": round(float(final[c]) * 100, 1),
                "Why": self._why(toks, b) if b is not None else "",
                "Matched rows": int(matches[c]),
                "_row": b,
            })
        out = pd.DataFrame(rows)
        if self.o_raw is not None and len(out):
            avg = []
            for c in order[:n]:
                sel = (codes == c) & (sims > 0) & valid & ~np.isnan(self.o_raw)
                avg.append(round(float(self.o_raw[sel].mean()), 2) if sel.any() else np.nan)
            out[f"Avg {self.cfg.outcome_col}"] = avg
        out.attrs["note"] = note
        return out

    def _mmr(self, order, final, best, n, lam):
        pool = order[: max(3 * n, 20)]
        rows = [best[c] for c in pool]
        sim = (self.M[rows] @ self.M[rows].T).toarray()
        chosen: list[int] = []
        while len(chosen) < min(n, len(pool)):
            scores = []
            for j in range(len(pool)):
                if j in chosen:
                    scores.append(-np.inf)
                    continue
                redundancy = max((sim[j, k] for k in chosen), default=0.0)
                scores.append(lam * final[pool[j]] - (1 - lam) * redundancy)
            chosen.append(int(np.argmax(scores)))
        return [pool[j] for j in chosen]

    # ---------------------------------------------- 2. "students like me" kNN
    def profile_vector(self, profile: dict, free_text: str = "") -> sp.csr_matrix:
        parts = []
        for key in self.block_order:
            w = np.sqrt(self.block_w[key])
            if key == "num":
                z = np.zeros((1, len(self.num_feats)))
                for j, col in enumerate(self.num_feats):
                    v = profile.get(col)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                    if not np.isnan(v):
                        z[0, j] = (v - self.num_mean[col]) / self.num_std[col]
                parts.append(sp.csr_matrix(self._num_embed(np.clip(z, -3, 3))) * w)
            elif key == "cat":
                rows = []
                for col, cv, weight in self.cat_vecs:
                    v = profile.get(col)
                    if v is None or str(v).strip() in ("", "(any)", "nan"):
                        rows.append(sp.csr_matrix((1, len(cv.vocabulary_))))
                    else:
                        rows.append(cv.transform([str(v)]) * weight)
                x = normalize(sp.hstack(rows, format="csr")) if rows else sp.csr_matrix((1, 0))
                parts.append(x * w)
            elif key == "text":
                bits = []
                for col in self.cfg.interest_cols:
                    v = profile.get(col)
                    if v not in (None, "", "(any)"):
                        bits += self._cell_tokens(v, self.multi.get(col, False))
                bits += self._cell_tokens(free_text, False, drop_stop=True)
                q = (self.tf_word.transform([" ".join(bits)])
                     if bits and self.tf_word is not None
                     else sp.csr_matrix((1, self.D_word.shape[1])))
                parts.append(q * w)
        return sp.hstack(parts, format="csr").astype(np.float32)

    @staticmethod
    def _vote(sims, codes, o_pct, n_classes, k, power=2.0):
        k = max(1, min(k, len(sims)))
        nn = np.argpartition(-sims, k - 1)[:k]
        nn = nn[(sims[nn] > 0) & (codes[nn] >= 0)]
        if len(nn) == 0:
            return np.zeros(n_classes), np.zeros(n_classes, dtype=int), nn
        w = (sims[nn] ** power) * (0.75 + 0.5 * o_pct[nn])
        sc = np.bincount(codes[nn], weights=w, minlength=n_classes)
        cnt = np.bincount(codes[nn], minlength=n_classes)
        return sc, cnt, nn

    def _knn_frame(self, sims, target, n, k, exclude_value=None):
        codes, cats = self._codes(target)
        C = len(cats)
        sc, cnt, nn = self._vote(sims, codes, self.o_pct, C, k)
        if exclude_value is not None and exclude_value in cats:
            sc[cats.index(str(exclude_value))] = 0.0
        note = None
        if sc.sum() <= 0:
            freq = np.bincount(codes[codes >= 0], minlength=C).astype(float)
            sc, note = freq, "No close matches - showing the most common choices."
        order = [c for c in np.argsort(-sc) if sc[c] > 0][:n]
        total = sc.sum()
        singletons = len(nn) > 0 and cnt.max() <= 1
        best_sim = np.zeros(C)
        if singletons:
            np.maximum.at(best_sim, codes[nn], sims[nn])
        rows = []
        for rank, c in enumerate(order, 1):
            share = 100 * best_sim[c] if singletons else 100 * sc[c] / total
            row = {"Rank": rank, target: cats[c], "Match %": round(float(np.clip(share, 0, 100)), 1),
                   "Similar profiles": int(cnt[c])}
            if self.o_raw is not None:
                sel = nn[codes[nn] == c]
                vals = self.o_raw[sel]
                vals = vals[~np.isnan(vals)]
                row[f"Avg {self.cfg.outcome_col}"] = round(float(vals.mean()), 2) if len(vals) else np.nan
            rows.append(row)
        out = pd.DataFrame(rows)
        out.attrs["note"], out.attrs["neighbours"] = note, nn
        return out

    def recommend_for_record(self, i: int, target: str, n=5, k=25, exclude_current=False):
        sims = (self.M @ self.M[i].T).toarray().ravel()
        sims[i] = -1
        current = self.df.at[i, target] if exclude_current else None
        out = self._knn_frame(sims, target, n, k, exclude_value=current)
        out.attrs["sims"] = sims
        return out

    def recommend_for_profile(self, profile: dict, target: str, n=5, k=25, free_text=""):
        q = self.profile_vector(profile, free_text)
        sims = (self.M @ q.T).toarray().ravel()
        out = self._knn_frame(sims, target, n, k)
        out.attrs["sims"] = sims
        return out

    def recommend_from_rows(self, idxs, target: str, n=5, k=25):
        """'Because you liked these rows' - average the chosen rows and look for neighbours."""
        idxs = [int(i) for i in idxs]
        q = sp.csr_matrix(self.M[idxs].mean(axis=0))
        sims = (self.M @ q.T).toarray().ravel()
        sims[idxs] = -1
        out = self._knn_frame(sims, target, n, k)
        out.attrs["sims"] = sims
        return out

    def similar_rows(self, sims: np.ndarray, k: int = 5) -> pd.DataFrame:
        idx = np.argsort(-sims)[:k]
        out = self.df.loc[idx].copy()
        out.insert(0, "Similarity %", np.round(100 * np.clip(sims[idx], 0, 1), 1))
        return out

    # --------------------------------------------------------- 3. batch mode
    def recommend_all(self, targets=None, n=3, k=25, exclude_current=False, max_rows=20000, progress=None):
        targets = list(targets or self.cfg.targets)
        total = min(self.n, max_rows)
        base = pd.DataFrame({"Row": np.arange(1, total + 1)})
        if self.cfg.label_col:
            base.insert(0, self.cfg.label_col, self.df[self.cfg.label_col].iloc[:total].to_numpy())
        results = {t: [] for t in targets}
        conf = {t: [] for t in targets}
        step = 500
        for start in range(0, total, step):
            idx = np.arange(start, min(start + step, total))
            S = (self.M[idx] @ self.M.T).toarray()
            S[np.arange(len(idx)), idx] = -1
            for t in targets:
                codes, cats = self._codes(t)
                for r, i in enumerate(idx):
                    sc, _, _ = self._vote(S[r], codes, self.o_pct, len(cats), k)
                    if exclude_current and codes[i] >= 0:
                        sc[codes[i]] = 0
                    order = [c for c in np.argsort(-sc) if sc[c] > 0][:n]
                    names = [cats[c] for c in order] + [""] * (n - len(order))
                    results[t].append(names)
                    conf[t].append(round(100 * sc[order[0]] / sc.sum(), 1) if order else np.nan)
            if progress:
                progress(min(1.0, (start + step) / total))
        ordinals = ["1st", "2nd", "3rd", "4th", "5th"]
        for t in targets:
            arr = results[t]
            for j in range(n):
                base[f"{t} - {ordinals[j] if j < 5 else f'{j + 1}th'} choice"] = [row[j] for row in arr]
            base[f"{t} - confidence %"] = conf[t]
        return base

    # ----------------------------------------------------------- 4. evaluation
    def evaluate(self, target: str, k=25, top_n=3, test_size=0.2, seed=42, max_test=2000):
        codes, cats = self._codes(target)
        C = len(cats)
        valid = np.where(codes >= 0)[0]
        if len(valid) < 30 or C < 2 or C > len(valid) / 2:
            return None
        rng = np.random.default_rng(seed)
        perm = rng.permutation(valid)
        n_test = max(1, min(int(len(valid) * test_size), max_test))
        test, train = perm[:n_test], perm[n_test:]
        S = (self.M[test] @ self.M[train].T).toarray()
        ctr, otr = codes[train], self.o_pct[train]
        freq = np.bincount(ctr, minlength=C).astype(float)
        pop_rank = {c: r for r, c in enumerate(np.argsort(-freq), 1)}

        def metrics(ranks):
            ranks = np.asarray(ranks)
            return {"hit1": float((ranks == 1).mean()), "hitn": float((ranks <= top_n).mean()),
                    "mrr": float((1.0 / ranks).mean())}

        model_ranks, base_ranks, preds = [], [], []
        for r, i in enumerate(test):
            sc, _, _ = self._vote(S[r], ctr, otr, C, k)
            adj = sc + 1e-6 * freq / max(freq.max(), 1)
            order = np.argsort(-adj)
            true = codes[i]
            model_ranks.append(int(np.where(order == true)[0][0]) + 1)
            base_ranks.append(pop_rank[true])
            preds.append(order[0])
        truth = codes[test]
        confusion = pd.crosstab(pd.Series([cats[c] for c in truth], name="Actual"),
                                pd.Series([cats[c] for c in preds], name="Predicted (top-1)"))
        per_class = (
            pd.DataFrame({"truth": truth, "pred": preds})
            .assign(hit=lambda d: d.truth == d.pred)
            .groupby("truth").agg(Support=("hit", "size"), Recall=("hit", "mean"))
        )
        per_class.index = [cats[c] for c in per_class.index]
        per_class["Recall"] = per_class["Recall"].round(3)
        return {
            "target": target, "n_test": int(n_test), "n_train": int(len(train)), "classes": C,
            "top_n": top_n, "model": metrics(model_ranks), "baseline": metrics(base_ranks),
            "random_hit1": 1.0 / C, "confusion": confusion, "per_class": per_class,
        }

    # -------------------------------------------------------------- 5. guidance
    def guidance(self, record: dict) -> list[str]:
        tips: list[str] = []
        oc = self.cfg.outcome_col
        if oc and self.o_raw is not None:
            try:
                v = float(record.get(oc))
                valid = self.o_raw[~np.isnan(self.o_raw)]
                pct = float((valid <= v).mean())
                if pct < 0.33:
                    tips.append(f"**{oc}: {v:g}** is in the lower third of the dataset ({pct:.0%} percentile) - "
                                "start with foundational, guided material and short practice sets before moving on.")
                elif pct < 0.67:
                    tips.append(f"**{oc}: {v:g}** is mid-range ({pct:.0%} percentile) - mix revision with "
                                "applied practice and gradually raise difficulty.")
                else:
                    tips.append(f"**{oc}: {v:g}** is in the top third ({pct:.0%} percentile) - stretch with "
                                "advanced, project-style or challenge material.")
            except (TypeError, ValueError):
                pass
        for col in self.num_feats:
            if col == oc or not _has_hint(col, BEHAVIOUR_HINTS):
                continue
            try:
                v = float(record.get(col))
            except (TypeError, ValueError):
                continue
            s = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if s.empty or np.isnan(v):
                continue
            pct, med = float((s <= v).mean()), float(s.median())
            if pct < 0.25:
                tips.append(f"**{col}: {v:g}** is low compared with the typical {med:g} - raising it is the "
                            "quickest lever for improvement.")
            elif pct > 0.75:
                tips.append(f"**{col}: {v:g}** is a strength (typical: {med:g}) - keep the routine going.")
        return tips
