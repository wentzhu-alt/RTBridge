from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
import json
import math
import re
import zipfile
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d, UnivariateSpline, CubicSpline, Akima1DInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neighbors import KNeighborsRegressor

# Broad RT-mapping model library. The first six are the manuscript models.
EXPANDED_MODEL_NAMES = [
    "Linear regression",
    "Quadratic regression",
    "Cubic regression",
    "Ridge cubic regression",
    "RANSAC linear",
    "Huber linear",
    "Theil-Sen linear",
    "Piecewise linear",
    "Cubic spline",
    "Smoothing spline/GAM",
    "LOESS",
    "PCHIP",
    "Akima spline",
    "Isotonic regression",
    "Monotone GAM",
    "SVR-RBF",
    "Gaussian process",
    "Random forest",
    "Extra trees",
    "Gradient boosting",
    "KNN regression",
]
DEFAULT_MODEL_NAMES = [
    "Linear regression",
    "Piecewise linear",
    "LOESS",
    "PCHIP",
    "Smoothing spline/GAM",
    "Monotone GAM",
]

MODEL_GROUPS = {
    "Manuscript default": DEFAULT_MODEL_NAMES,
    "Classic regression": ["Linear regression", "Quadratic regression", "Cubic regression", "Ridge cubic regression"],
    "Robust regression": ["RANSAC linear", "Huber linear", "Theil-Sen linear"],
    "Interpolation / local warping": ["Piecewise linear", "Cubic spline", "PCHIP", "Akima spline"],
    "Smooth / monotone": ["LOESS", "Smoothing spline/GAM", "Isotonic regression", "Monotone GAM"],
    "Machine learning": ["SVR-RBF", "Gaussian process", "Random forest", "Extra trees", "Gradient boosting", "KNN regression"],
    "All available": EXPANDED_MODEL_NAMES,
}

MODEL_NOTES = {
    "Linear regression": "Global linear baseline.",
    "Quadratic regression": "Global polynomial baseline with one curvature term.",
    "Cubic regression": "Global polynomial baseline with higher flexibility.",
    "Ridge cubic regression": "Regularized polynomial fit.",
    "RANSAC linear": "Robust linear fit that down-weights large outliers.",
    "Huber linear": "Robust linear fit using Huber loss.",
    "Theil-Sen linear": "Robust median-slope linear estimator.",
    "Piecewise linear": "Anchor-to-anchor linear interpolation.",
    "Cubic spline": "Interpolating cubic spline.",
    "Smoothing spline/GAM": "Cubic smoothing spline used as a GAM-like RT trend.",
    "LOESS": "Local weighted linear smoother.",
    "PCHIP": "Shape-preserving piecewise cubic interpolation.",
    "Akima spline": "Local cubic interpolation robust to oscillation from abrupt changes.",
    "Isotonic regression": "Monotone non-decreasing stepwise fit.",
    "Monotone GAM": "Isotonic trend with light cubic-spline smoothing.",
    "SVR-RBF": "Kernel support-vector regression.",
    "Gaussian process": "Probabilistic smooth nonlinear regression.",
    "Random forest": "Tree ensemble nonlinear regression.",
    "Extra trees": "Extremely randomized tree ensemble regression.",
    "Gradient boosting": "Boosted decision-tree nonlinear regression.",
    "KNN regression": "Local nearest-neighbor regression.",
}

# Built-in biological-summary behavior used by all RTBridge calculations.
# Technical replicate injections are grouped within each independent biological sample
# and averaged before correlation, ratio error, ratio score, and ACS are calculated.
# The replicate count is inferred from column names or supplied as a dataset-level
# recognition aid; replicate averaging is never disabled.
BIOLOGICAL_SUMMARY_METHOD = "mean"
REQUIRE_POSITIVE_CORRELATION = True

@dataclass
class RTBridgeSettings:
    mass_ppm_unique: float = 20.0
    mass_ppm_anchor: float = 10.0
    mass_ppm_candidate: float = 10.0
    qc_mean_min: float = 100.0
    qc_cv_max: float = 0.30
    acs_min: float = 0.50
    correlation_weight: float = 0.50
    ratio_weight: float = 0.50
    square_correlation: bool = False
    rt_window_min: float = 0.35
    verification_mz_ppm: float = 10.0
    verification_rt5_min: float = 0.20
    verification_rt25_min: float = 0.20
    gam_screen_iterations: int = 2
    residual_multiplier: float = 2.0
    output_dpi: int = 600
    random_seed: int = 1
    selected_models: List[str] = field(default_factory=lambda: DEFAULT_MODEL_NAMES.copy())
    export_all_candidate_pairs: bool = True
    make_model_fit_diagnostics: bool = True

@dataclass
class DatasetSpec:
    dataset: str
    matrix: str
    mode: str
    source_file: str
    target_file: str
    technical_replicates: Optional[int] = None
    expected_biological_samples: Optional[int] = None

class ColumnError(ValueError):
    pass

# --------------------------- file/column utilities ---------------------------
def read_table_auto(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/TXT/Excel with Windows-friendly encoding fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1", "gb18030", "gbk", "utf-16"]
    errors = []
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError as e:
            errors.append(f"{enc}: {e}")
        except pd.errors.ParserError as e:
            try:
                return pd.read_csv(path, encoding=enc, sep=None, engine="python", on_bad_lines="skip")
            except Exception as e2:
                errors.append(f"{enc}: parser error {e}; retry failed: {e2}")
        except Exception as e:
            errors.append(f"{enc}: {e}")
    raise ValueError("Could not read file with common encodings/delimiters: " + str(path) + "\n" + "\n".join(errors[-5:]))

def ppm_error(a, b):
    return 1e6 * np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) / np.asarray(b, dtype=float)

def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool=True, role: str="column") -> Optional[str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        lc = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    if required:
        raise ColumnError(f"Could not find {role}. Tried: {list(candidates)}. Available columns: {list(df.columns)}")
    return None

def detect_columns(df: pd.DataFrame) -> dict:
    mz = find_column(df, ["mz", "m/z", "mass", "mass_to_charge", "mass to charge"], role="m/z column")
    rt = find_column(df, ["rt", "retention time", "retention_time", "retentiontime"], role="RT column")
    id_col = find_column(df, ["feature_id", "featureid", "id", "row id", "compound", "name"], required=False)
    # Convert numeric-like text columns before detecting abundance.
    numeric_cols = []
    for c in df.columns:
        if c in [mz, rt]:
            numeric_cols.append(c)
        else:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= max(3, int(0.20 * len(df))):
                df[c] = conv
                numeric_cols.append(c)
    excluded_numeric = [
        c for c in numeric_cols
        if c not in [mz, rt] and (
            "cv" in str(c).strip().lower()
            or str(c).strip().lower() in {"row", "index", "rank"}
        )
    ]
    abundance_cols = [c for c in numeric_cols if c not in [mz, rt] and c not in excluded_numeric]
    qc_cols = [c for c in abundance_cols if "qc" in str(c).lower() or "pool" in str(c).lower()]
    bio_cols = [c for c in abundance_cols if c not in qc_cols]
    if not abundance_cols:
        raise ColumnError("No numeric abundance columns detected. Please keep sample/QC abundance columns numeric.")
    if not qc_cols:
        warnings.warn("No QC columns detected by name. QC filtering will be skipped and all numeric columns will be used for abundance scoring.")
    return {
        "mz": mz, "rt": rt, "id": id_col, "abundance": abundance_cols,
        "qc": qc_cols, "bio": bio_cols, "excluded_numeric": excluded_numeric,
    }

def _biological_sample_group_key(column_name: str) -> Optional[Tuple[str, int]]:
    """Infer biological-sample identity and technical-injection number.

    Supported examples include:
    P1-1-NEG, P1_2_NEG, SampleA-rep3, SampleA_injection_4_POS,
    Plasma_S01_R5, and 25Plasma-S01-techrep-2-NEG.
    """
    name = str(column_name).strip()
    mode_suffix = r"(?:[\s._-]+(?:\d*(?:POS|NEG)))?"
    patterns = [
        # Explicit replicate/injection/run labels are preferred because they avoid
        # confusing digits inside the biological sample identifier with replicate IDs.
        rf"^(?P<sample>.+?)[\s._-]+(?:tech(?:nical)?[\s._-]*rep(?:licate)?|rep(?:licate)?|inj(?:ection)?|run|r)[\s._-]*(?P<rep>\d+){mode_suffix}$",
        # Publication-style names such as P1-1-NEG or 25P1_3_POS.
        rf"^(?P<sample>.+?)[\s._-]+(?P<rep>\d+){mode_suffix}$",
    ]
    for pattern in patterns:
        match = re.match(pattern, name, flags=re.IGNORECASE)
        if match:
            sample = re.sub(r"[\s._-]+$", "", match.group("sample")).strip()
            if sample:
                return sample, int(match.group("rep"))
    return None


def _canonical_biological_sample_id(sample_id: str) -> str:
    """Normalize harmless source/target method prefixes before alignment checks."""
    value = str(sample_id).strip()
    value = re.sub(r"^(?:source|target|short|long)[\s._-]+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:5|15|25)(?:min)?[\s._-]*(?=[A-Za-z])", "", value, flags=re.IGNORECASE)
    return value.casefold()


def _format_replicate_counts(counts: List[int]) -> str:
    if not counts:
        return ""
    unique = sorted(set(int(x) for x in counts))
    if len(unique) == 1:
        return str(unique[0])
    return ",".join(str(x) for x in unique)


def _infer_biological_sample_groups(
    raw_columns: List[str],
    expected_replicates: Optional[int] = None,
    expected_samples: Optional[int] = None,
) -> Tuple[List[Tuple[str, List[str]]], str]:
    """Group technical injections within independent biological samples.

    Grouping priority:
    1. Automatic name-based recognition with any replicate count.
    2. Dataset-level fixed replicate count using consecutive sample-major columns.

    RTBridge never silently assumes three injections. If names are not informative,
    the user must provide the number of technical injections per biological sample.
    """
    if expected_replicates is not None and int(expected_replicates) < 1:
        raise ColumnError("Technical injections per biological sample must be at least 1.")
    if expected_samples is not None and int(expected_samples) < 1:
        raise ColumnError("Expected biological sample count must be at least 1.")

    parsed = [_biological_sample_group_key(c) for c in raw_columns]
    if raw_columns and all(item is not None for item in parsed):
        order: List[str] = []
        grouped: Dict[str, List[Tuple[int, str]]] = {}
        display_names: Dict[str, str] = {}
        for column, parsed_item in zip(raw_columns, parsed):
            assert parsed_item is not None
            sample, replicate = parsed_item
            key = sample.casefold()
            if key not in grouped:
                grouped[key] = []
                display_names[key] = sample
                order.append(key)
            grouped[key].append((replicate, column))

        candidate: List[Tuple[str, List[str]]] = []
        for key in order:
            rows = sorted(grouped[key], key=lambda x: x[0])
            replicate_ids = [replicate for replicate, _ in rows]
            if len(set(replicate_ids)) != len(rows):
                raise ColumnError(
                    f"Duplicate replicate/injection numbers were detected for biological sample "
                    f"'{display_names[key]}': {replicate_ids}."
                )
            candidate.append((display_names[key], [column for _, column in rows]))

        replicate_mismatch = (
            expected_replicates is not None
            and any(len(columns) != int(expected_replicates) for _, columns in candidate)
        )
        sample_mismatch = expected_samples is not None and len(candidate) != int(expected_samples)
        ambiguous_single_group = len(candidate) == 1 and expected_samples != 1

        if not replicate_mismatch and not sample_mismatch and not ambiguous_single_group:
            return candidate, "automatic_name_based"

        # A fixed replicate count explicitly selected by the user takes precedence when
        # generic column names such as Abundance_1 ... Abundance_50 look like one giant
        # replicate series. Fall through to safe consecutive grouping when dimensions fit.
        if expected_replicates is None:
            detail = {sample: len(columns) for sample, columns in candidate}
            raise ColumnError(
                "Column names were parseable but did not define an unambiguous set of "
                f"independent biological samples. Recognized groups: {detail}. Rename columns "
                "to include both sample and replicate identifiers, or select the technical "
                "injection count in the dataset panel."
            )

    if expected_replicates is None:
        unparsed = [str(c) for c, item in zip(raw_columns, parsed) if item is None]
        raise ColumnError(
            "RTBridge could not unambiguously recognize technical-replicate groups from "
            "the biological abundance column names. Use names such as Sample01_rep1, "
            "Sample01_rep2, ... or select the number of technical injections per biological "
            f"sample in the dataset panel. Unrecognized columns include: {unparsed[:12]}"
        )

    reps = int(expected_replicates)
    if len(raw_columns) % reps != 0:
        raise ColumnError(
            f"Detected {len(raw_columns)} biological injection columns, which cannot be divided "
            f"into consecutive sample-major groups of {reps}. Check the replicate count, remove "
            "metadata columns, or rename columns to include sample and replicate identifiers."
        )
    n_samples = len(raw_columns) // reps
    if expected_samples is not None and n_samples != int(expected_samples):
        raise ColumnError(
            f"Expected {int(expected_samples)} independent biological samples × {reps} injections "
            f"= {int(expected_samples) * reps} biological injection columns, but found "
            f"{len(raw_columns)} columns ({n_samples} groups)."
        )
    groups = [
        (f"Sample_{index + 1}", raw_columns[index * reps:(index + 1) * reps])
        for index in range(n_samples)
    ]
    return groups, f"fixed_count_consecutive_{reps}"


def _summarize_biological_replicates(
    df: pd.DataFrame,
    cols: dict,
    table_name: str,
    expected_replicates: Optional[int] = None,
    expected_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Create one mean abundance vector per independent biological sample."""
    raw = list(cols.get("bio", []))
    if not raw:
        raise ColumnError(f"No biological abundance columns were detected in {table_name}.")

    inferred_groups, grouping_method = _infer_biological_sample_groups(
        raw,
        expected_replicates=expected_replicates,
        expected_samples=expected_samples,
    )
    summary_cols: List[str] = []
    groups: List[dict] = []
    sample_ids: List[str] = []
    replicate_counts: List[int] = []

    for sample_index, (sample_id, group) in enumerate(inferred_groups, start=1):
        out_col = f"__biological_sample_{sample_index}_mean"
        numeric = df[group].apply(pd.to_numeric, errors="coerce")
        df[out_col] = numeric.mean(axis=1, skipna=True)
        summary_cols.append(out_col)
        sample_ids.append(str(sample_id))
        replicate_counts.append(len(group))
        groups.append({
            "sample_index": sample_index,
            "sample_id": str(sample_id),
            "summary_column": out_col,
            "raw_injection_columns": group,
            "number_of_injections": len(group),
            "summary_method": BIOLOGICAL_SUMMARY_METHOD,
            "grouping_method": grouping_method,
        })

    cols = dict(cols)
    cols["bio_raw"] = raw
    cols["bio"] = summary_cols
    cols["biological_sample_groups"] = groups
    cols["biological_sample_ids"] = sample_ids
    # Compatibility aliases used by older internal functions; values now refer to
    # general independent biological samples, not necessarily human donors.
    cols["donor_groups"] = groups
    cols["donor_ids"] = sample_ids
    cols["n_donors"] = len(summary_cols)
    cols["n_biological_samples"] = len(summary_cols)
    cols["replicate_counts"] = replicate_counts
    cols["replicates_per_donor"] = replicate_counts[0] if len(set(replicate_counts)) == 1 else 0
    cols["replicate_count_summary"] = _format_replicate_counts(replicate_counts)
    cols["replicate_grouping_method"] = grouping_method
    cols["biological_summary"] = BIOLOGICAL_SUMMARY_METHOD
    return df, cols

def read_feature_table(
    path: Path,
    settings: RTBridgeSettings,
    table_name: str,
    expected_replicates: Optional[int] = None,
    expected_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    df = read_table_auto(path)
    cols = detect_columns(df)
    df = df.copy()
    df[cols["mz"]] = pd.to_numeric(df[cols["mz"]], errors="coerce")
    df[cols["rt"]] = pd.to_numeric(df[cols["rt"]], errors="coerce")
    for c in cols["abundance"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df, cols = _summarize_biological_replicates(
        df, cols, table_name,
        expected_replicates=expected_replicates,
        expected_samples=expected_samples,
    )
    df = df.dropna(subset=[cols["mz"], cols["rt"]]).reset_index(drop=True)
    df["__row_id"] = np.arange(len(df))
    df["__mz"] = df[cols["mz"]].astype(float)
    df["__rt"] = df[cols["rt"]].astype(float)
    qc_cols = cols["qc"]
    if qc_cols:
        df["__qc_mean"] = df[qc_cols].mean(axis=1, skipna=True)
        if len(qc_cols) >= 2:
            mu = df[qc_cols].mean(axis=1, skipna=True)
            sd = df[qc_cols].std(axis=1, skipna=True, ddof=1)
            df["__qc_cv"] = sd / mu.replace(0, np.nan)
            df = df[(df["__qc_cv"].isna()) | (df["__qc_cv"] <= settings.qc_cv_max)].copy()
        else:
            df["__qc_cv"] = np.nan
    else:
        df["__qc_mean"] = df[cols["abundance"]].mean(axis=1, skipna=True)
        df["__qc_cv"] = np.nan
    df["__table"] = table_name
    return df.reset_index(drop=True), cols

# --------------------------- anchor and scoring ---------------------------
def mark_unique_by_mz(df: pd.DataFrame, ppm: float) -> np.ndarray:
    mz = df["__mz"].to_numpy(float)
    order = np.argsort(mz)
    mz_sorted = mz[order]
    unique = np.ones(len(mz), dtype=bool)
    left = 0
    for right in range(len(mz_sorted)):
        while left < right and ppm_error(mz_sorted[right], mz_sorted[left]) > ppm:
            left += 1
        if right - left > 0:
            unique[order[left:right+1]] = False
    return unique

def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r across independent biological-sample-level observations."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan
    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return r if np.isfinite(r) else np.nan

def positive_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Backward-compatible positive contribution used inside ACS."""
    r = pearson_corr(x, y)
    return max(0.0, r) if np.isfinite(r) else 0.0

def _normalized_acs_weights(settings: RTBridgeSettings | None = None) -> Tuple[float, float]:
    """Return correlation and ratio weights normalized to sum to one."""
    wc = 0.50 if settings is None else float(settings.correlation_weight)
    wr = 0.50 if settings is None else float(settings.ratio_weight)
    if not np.isfinite(wc) or not np.isfinite(wr) or wc < 0 or wr < 0 or (wc + wr) <= 0:
        raise ValueError("ACS weights must be finite, non-negative, and have a positive sum.")
    total = wc + wr
    return wc / total, wr / total

def acs_for_pair(src_row: pd.Series, tgt_row: pd.Series, src_bio: List[str], tgt_bio: List[str], settings: RTBridgeSettings | None = None) -> Tuple[float, float, float]:
    n = min(len(src_bio), len(tgt_bio))
    if n == 0:
        return 0.0, 0.0, 0.0
    xs = src_row[src_bio[:n]].to_numpy(dtype=float)
    ys = tgt_row[tgt_bio[:n]].to_numpy(dtype=float)
    src_qc = float(src_row.get("__qc_mean", np.nan))
    tgt_qc = float(tgt_row.get("__qc_mean", np.nan))
    return _acs_from_arrays(xs, ys, src_qc, tgt_qc, settings)

def _acs_from_arrays(xs: np.ndarray, ys: np.ndarray, src_qc: float, tgt_qc: float, settings: RTBridgeSettings | None = None) -> Tuple[float, float, float]:
    """Calculate ACS end-to-end from independent biological-sample summaries."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask_corr = np.isfinite(xs) & np.isfinite(ys)
    if mask_corr.sum() < 3:
        corr_raw = np.nan
    else:
        x = xs[mask_corr]; y = ys[mask_corr]
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            corr_raw = np.nan
        else:
            corr_raw = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(corr_raw):
                corr_raw = np.nan
    # Mandatory biological-direction filter: undefined, zero, and negative
    # correlations are ineligible before any linear or squared transformation.
    corr_is_positive = bool(np.isfinite(corr_raw) and corr_raw > 0.0)
    corr_positive = float(corr_raw) if corr_is_positive else 0.0
    corr_component = (
        corr_positive ** 2
        if corr_is_positive and settings is not None and settings.square_correlation
        else corr_positive
    )
    if not np.isfinite(src_qc) or not np.isfinite(tgt_qc) or src_qc <= 0 or tgt_qc <= 0:
        ratio_score = 0.0
    else:
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
        if mask.sum() == 0:
            ratio_score = 0.0
        else:
            r = (ys[mask] / tgt_qc) / (xs[mask] / src_qc)
            ratio_error = float(np.nanmedian(np.abs(np.log2(r)))) if len(r) else np.inf
            ratio_score = 1.0 / (1.0 + ratio_error) if np.isfinite(ratio_error) else 0.0
    # Return an ineligible ACS for nonpositive correlations even when the ratio
    # weight is large. Callers also enforce corr_score > 0 as a hard filter.
    if not corr_is_positive:
        return 0.0, float(corr_raw) if np.isfinite(corr_raw) else np.nan, float(ratio_score)
    wc, wr = _normalized_acs_weights(settings)
    acs = wc * corr_component + wr * ratio_score
    return float(acs), float(corr_raw), float(ratio_score)

def mz_candidates(src_mz: float, target_df: pd.DataFrame, ppm: float) -> pd.DataFrame:
    err = ppm_error(target_df["__mz"].to_numpy(float), src_mz)
    return target_df.loc[err <= ppm].copy()

def _mz_window_positions(mz_value: float, sorted_mz: np.ndarray, sorted_positions: np.ndarray, ppm: float) -> np.ndarray:
    """Return target row positions within a symmetric ppm window, using searchsorted."""
    if not np.isfinite(mz_value):
        return np.array([], dtype=int)
    delta = ppm / 1e6
    lo = mz_value * (1.0 - delta)
    hi = mz_value * (1.0 + delta)
    left = np.searchsorted(sorted_mz, lo, side="left")
    right = np.searchsorted(sorted_mz, hi, side="right")
    if right <= left:
        return np.array([], dtype=int)
    return sorted_positions[left:right]

def build_anchors(src: pd.DataFrame, tgt: pd.DataFrame, src_cols: dict, tgt_cols: dict, settings: RTBridgeSettings) -> pd.DataFrame:
    """Build the same reciprocal unique-anchor set using indexed m/z lookup.

    Scientific filters and output ordering are unchanged. The speed improvement
    comes only from replacing repeated full-table scans with binary searches.
    """
    s, t = src.copy(), tgt.copy()
    s["__unique"] = mark_unique_by_mz(s, settings.mass_ppm_unique)
    t["__unique"] = mark_unique_by_mz(t, settings.mass_ppm_unique)
    su = s[s["__unique"] & (s["__qc_mean"] > settings.qc_mean_min)].copy().reset_index(drop=True)
    tu = t[t["__unique"] & (t["__qc_mean"] > settings.qc_mean_min)].copy().reset_index(drop=True)
    empty_cols = ["Source_row","Target_row","Source_mz","Target_mz","Source_RT","Target_RT","Mass_error_ppm","Source_QC_mean","Target_QC_mean","ACS","corr_score","ratio_score"]
    if su.empty or tu.empty:
        return pd.DataFrame(columns=empty_cols)

    su_mz = su["__mz"].to_numpy(float)
    tu_mz = tu["__mz"].to_numpy(float)
    su_order = np.argsort(su_mz)
    tu_order = np.argsort(tu_mz)
    su_sorted = su_mz[su_order]
    tu_sorted = tu_mz[tu_order]

    n_bio = min(len(src_cols.get("bio", [])), len(tgt_cols.get("bio", [])))
    su_bio = su[src_cols.get("bio", [])[:n_bio]].to_numpy(float) if n_bio else np.empty((len(su), 0), dtype=float)
    tu_bio = tu[tgt_cols.get("bio", [])[:n_bio]].to_numpy(float) if n_bio else np.empty((len(tu), 0), dtype=float)
    su_qc = su["__qc_mean"].to_numpy(float)
    tu_qc = tu["__qc_mean"].to_numpy(float)

    # Forward eligibility: the source feature must have exactly one target inside
    # the original anchor ppm definition, whose denominator is source m/z.
    forward = np.full(len(su), -1, dtype=int)
    for i, mz in enumerate(su_mz):
        pos = _mz_window_positions(float(mz), tu_sorted, tu_order, settings.mass_ppm_anchor)
        if pos.size == 1:
            forward[i] = int(pos[0])

    rows = []
    for i, j in enumerate(forward):
        if j < 0:
            continue
        # Reciprocal check uses target m/z as the denominator, exactly as before.
        back = _mz_window_positions(float(tu_mz[j]), su_sorted, su_order, settings.mass_ppm_anchor)
        if back.size != 1 or int(back[0]) != i:
            continue
        acs, corr, ratio = _acs_from_arrays(su_bio[i], tu_bio[j], su_qc[i], tu_qc[j], settings)
        if REQUIRE_POSITIVE_CORRELATION and (not np.isfinite(corr) or corr <= 0):
            continue
        if acs < settings.acs_min:
            continue
        sr = su.iloc[i]
        tr = tu.iloc[j]
        rows.append({
            "Source_row": int(sr["__row_id"]), "Target_row": int(tr["__row_id"]),
            "Source_mz": float(su_mz[i]), "Target_mz": float(tu_mz[j]),
            "Source_RT": float(sr["__rt"]), "Target_RT": float(tr["__rt"]),
            "Mass_error_ppm": float(ppm_error(su_mz[i], tu_mz[j])),
            "Source_QC_mean": float(su_qc[i]), "Target_QC_mean": float(tu_qc[j]),
            "ACS": acs, "corr_score": corr, "ratio_score": ratio,
        })
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows).sort_values(["Source_RT", "Target_RT"]).reset_index(drop=True)

# --------------------------- RT model library ---------------------------
def _prepare_xy(x_train, y_train):
    df = pd.DataFrame({"x": np.asarray(x_train, float), "y": np.asarray(y_train, float)}).replace([np.inf, -np.inf], np.nan).dropna()
    df = df.groupby("x", as_index=False).mean().sort_values("x")
    return df["x"].to_numpy(float), df["y"].to_numpy(float)

def _poly_predict(x, y, xp, degree: int, ridge: bool=False):
    X = x.reshape(-1, 1); XP = xp.reshape(-1, 1)
    if ridge:
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
        model.fit(X, y)
        return model.predict(XP), {"degree": degree, "regularized": True}
    deg = min(degree, len(x)-1)
    coef = np.polyfit(x, y, deg)
    return np.polyval(coef, xp), {"degree": int(deg), "coefficients": [float(c) for c in coef]}

def fit_predict_model(model: str, x_train, y_train, x_pred, random_seed: int=1):
    x, y = _prepare_xy(x_train, y_train)
    xp = np.asarray(x_pred, float)
    if len(x) < 3:
        return np.full_like(xp, np.nan, dtype=float), {"error": "too few anchors"}
    X = x.reshape(-1, 1)
    XP = xp.reshape(-1, 1)
    try:
        if model == "Linear regression":
            slope, intercept = np.polyfit(x, y, 1)
            return slope * xp + intercept, {"slope": float(slope), "intercept": float(intercept)}
        if model == "Quadratic regression":
            return _poly_predict(x, y, xp, 2)
        if model == "Cubic regression":
            return _poly_predict(x, y, xp, 3)
        if model == "Ridge cubic regression":
            return _poly_predict(x, y, xp, 3, ridge=True)
        if model == "RANSAC linear":
            try:
                reg = RANSACRegressor(estimator=LinearRegression(), random_state=random_seed, min_samples=max(2, int(0.5*len(x))))
            except TypeError:  # older scikit-learn
                reg = RANSACRegressor(base_estimator=LinearRegression(), random_state=random_seed, min_samples=max(2, int(0.5*len(x))))
            reg.fit(X, y)
            return reg.predict(XP), {"inlier_fraction": float(np.mean(reg.inlier_mask_)) if hasattr(reg, "inlier_mask_") else np.nan}
        if model == "Huber linear":
            reg = HuberRegressor().fit(X, y)
            return reg.predict(XP), {"epsilon": float(reg.epsilon)}
        if model == "Theil-Sen linear":
            reg = TheilSenRegressor(random_state=random_seed).fit(X, y)
            return reg.predict(XP), {}
        if model == "Piecewise linear":
            return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")(xp), {}
        if model == "Cubic spline":
            if len(x) < 4:
                return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")(xp), {"fallback": "linear"}
            return CubicSpline(x, y, extrapolate=True)(xp), {}
        if model == "Smoothing spline/GAM":
            s_val = max(1e-6, 0.01 * len(x) * np.nanvar(y))
            spl = UnivariateSpline(x, y, s=s_val, k=min(3, len(x)-1))
            return spl(xp), {"method": "SciPy UnivariateSpline", "degree": int(min(3, len(x)-1)), "s": float(s_val)}
        if model == "LOESS":
            frac = max(0.08, min(0.50, 25 / max(len(x), 1)))
            k = max(5, int(math.ceil(frac * len(x))))
            out = np.empty_like(xp, dtype=float)
            for i, xx in enumerate(xp):
                dist = np.abs(x - xx)
                idx = np.argsort(dist)[:k]
                xs, ys = x[idx], y[idx]
                dmax = np.max(np.abs(xs - xx)) or 1.0
                w = (1 - (np.abs(xs - xx) / dmax) ** 3) ** 3
                try:
                    coef = np.polyfit(xs, ys, 1, w=w)
                    out[i] = coef[0] * xx + coef[1]
                except Exception:
                    out[i] = np.average(ys, weights=w)
            return out, {"frac": float(frac), "neighbors": int(k)}
        if model == "PCHIP":
            return PchipInterpolator(x, y, extrapolate=True)(xp), {}
        if model == "Akima spline":
            if len(x) < 5:
                return PchipInterpolator(x, y, extrapolate=True)(xp), {"fallback": "PCHIP"}
            return Akima1DInterpolator(x, y, method="akima", extrapolate=True)(xp), {}
        if model == "Isotonic regression":
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(x, y)
            return iso.predict(xp), {}
        if model == "Monotone GAM":
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            yy = iso.fit_transform(x, y)
            try:
                s_val = max(1e-6, 0.005 * len(x) * np.nanvar(yy))
                spl = UnivariateSpline(x, yy, s=s_val, k=min(3, len(x)-1))
                return spl(xp), {"method": "isotonic regression followed by SciPy UnivariateSpline", "degree": int(min(3, len(x)-1)), "s": float(s_val)}
            except Exception:
                return iso.predict(xp), {"method": "isotonic regression fallback"}
        if model == "SVR-RBF":
            reg = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.05))
            reg.fit(X, y)
            return reg.predict(XP), {"C": 10.0, "epsilon": 0.05}
        if model == "Gaussian process":
            kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.05)
            reg = GaussianProcessRegressor(kernel=kernel, random_state=random_seed, normalize_y=True, alpha=1e-6)
            # Limit very large fits for speed.
            if len(x) > 400:
                idx = np.linspace(0, len(x)-1, 400).astype(int)
                reg.fit(X[idx], y[idx])
            else:
                reg.fit(X, y)
            return reg.predict(XP), {"kernel": str(reg.kernel_)}
        if model == "Random forest":
            reg = RandomForestRegressor(n_estimators=80, random_state=random_seed, min_samples_leaf=2, n_jobs=-1)
            reg.fit(X, y)
            return reg.predict(XP), {"n_estimators": 80}
        if model == "Extra trees":
            reg = ExtraTreesRegressor(n_estimators=80, random_state=random_seed, min_samples_leaf=2, n_jobs=-1)
            reg.fit(X, y)
            return reg.predict(XP), {"n_estimators": 80}
        if model == "Gradient boosting":
            reg = GradientBoostingRegressor(random_state=random_seed, n_estimators=80, max_depth=2, learning_rate=0.05)
            reg.fit(X, y)
            return reg.predict(XP), {"n_estimators": 80}
        if model == "KNN regression":
            k = min(15, max(3, int(np.sqrt(len(x)))))
            reg = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance"))
            reg.fit(X, y)
            return reg.predict(XP), {"neighbors": int(k)}
    except Exception as e:
        return np.full_like(xp, np.nan, dtype=float), {"error": str(e)}
    raise ValueError(f"Unknown model: {model}")

def model_anchor_diagnostics(anchors: pd.DataFrame, model: str, workflow: str, dataset: str, matrix: str, mode: str, config: str, settings: RTBridgeSettings) -> pd.DataFrame:
    if anchors.empty or len(anchors) < 3:
        return pd.DataFrame()
    pred, params = fit_predict_model(model, anchors["Source_RT"].values, anchors["Target_RT"].values, anchors["Source_RT"].values, settings.random_seed)
    out = anchors[["Source_row", "Target_row", "Source_mz", "Target_mz", "Source_RT", "Target_RT", "ACS"]].copy()
    out["Predicted_Target_RT"] = pred
    out["Residual_min"] = out["Target_RT"] - out["Predicted_Target_RT"]
    out["Abs_residual_min"] = np.abs(out["Residual_min"])
    out["Dataset"] = dataset; out["Matrix"] = matrix; out["Mode"] = mode; out["Anchor_configuration"] = config
    out["Workflow"] = workflow; out["Model"] = model
    return out

# --------------------------- residual screening ---------------------------
def gam_based_screen(anchors: pd.DataFrame, settings: RTBridgeSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    current = anchors.copy().reset_index(drop=True)
    audit_rows = []
    if current.empty or len(current) < 8 or settings.gam_screen_iterations <= 0:
        return current, pd.DataFrame()
    for iteration in range(1, settings.gam_screen_iterations + 1):
        x = current["Source_RT"].to_numpy(float)
        y = current["Target_RT"].to_numpy(float)
        try:
            s_val = max(1e-6, 0.01 * len(x) * np.nanvar(y))
            spl = UnivariateSpline(*_prepare_xy(x, y), s=s_val, k=min(3, len(x)-1))
            fitted = spl(current["Source_RT"].to_numpy(float))
        except Exception:
            fitted, _ = fit_predict_model("PCHIP", x, y, current["Source_RT"].to_numpy(float), settings.random_seed)
        residual = np.abs(current["Target_RT"].to_numpy(float) - fitted)
        cutoff = settings.residual_multiplier * np.nanmean(residual)
        remove = residual > cutoff
        before = len(current)
        removed = int(np.nansum(remove))
        current = current.loc[~remove].copy().reset_index(drop=True)
        audit_rows.append({
            "Iteration": iteration, "Anchors_before": before, "Anchors_removed": removed, "Anchors_after": len(current),
            "Residual_cutoff_min": float(cutoff), "Median_abs_residual_min": float(np.nanmedian(residual)),
        })
        if removed == 0:
            break
    return current, pd.DataFrame(audit_rows)

# --------------------------- pairing and verification ---------------------------
def _select_candidate_pairs_direct(src: pd.DataFrame, tgt: pd.DataFrame, src_cols: dict, tgt_cols: dict,
                                   anchors: pd.DataFrame, model: str,
                                   settings: RTBridgeSettings) -> Tuple[pd.DataFrame, dict]:
    """Reference candidate-pair implementation, retained for one-model runs."""
    if anchors.empty or len(anchors) < 3:
        return pd.DataFrame(), {"error": "not enough anchors"}

    src = src.reset_index(drop=True)
    tgt = tgt.reset_index(drop=True)
    pred_rt, params = fit_predict_model(model, anchors["Source_RT"].values, anchors["Target_RT"].values, src["__rt"].values, settings.random_seed)

    src_mz = src["__mz"].to_numpy(float)
    src_rt = src["__rt"].to_numpy(float)
    src_row_id = src["__row_id"].to_numpy(int)
    src_qc = src["__qc_mean"].to_numpy(float) if "__qc_mean" in src.columns else np.full(len(src), np.nan)
    tgt_mz = tgt["__mz"].to_numpy(float)
    tgt_rt = tgt["__rt"].to_numpy(float)
    tgt_row_id = tgt["__row_id"].to_numpy(int)
    tgt_qc = tgt["__qc_mean"].to_numpy(float) if "__qc_mean" in tgt.columns else np.full(len(tgt), np.nan)
    sort_pos = np.argsort(tgt_mz)
    tgt_mz_sorted = tgt_mz[sort_pos]

    n_bio = min(len(src_cols.get("bio", [])), len(tgt_cols.get("bio", [])))
    src_bio_mat = src[src_cols["bio"][:n_bio]].to_numpy(float) if n_bio else np.empty((len(src), 0), dtype=float)
    tgt_bio_mat = tgt[tgt_cols["bio"][:n_bio]].to_numpy(float) if n_bio else np.empty((len(tgt), 0), dtype=float)

    rows = []
    for i in range(len(src)):
        prt = float(pred_rt[i]) if i < len(pred_rt) else np.nan
        if not np.isfinite(prt):
            continue
        cand_pos = _mz_window_positions(float(src_mz[i]), tgt_mz_sorted, sort_pos, settings.mass_ppm_candidate)
        if cand_pos.size == 0:
            continue
        cand_pos = cand_pos[np.abs(tgt_rt[cand_pos] - prt) <= float(settings.rt_window_min)]
        if cand_pos.size == 0:
            continue
        xs = src_bio_mat[i] if n_bio else np.array([], dtype=float)
        for j in cand_pos:
            ys = tgt_bio_mat[j] if n_bio else np.array([], dtype=float)
            acs, corr, ratio = _acs_from_arrays(xs, ys, src_qc[i], tgt_qc[j], settings)
            if REQUIRE_POSITIVE_CORRELATION and (not np.isfinite(corr) or corr <= 0):
                continue
            rows.append({
                "Source_row": int(src_row_id[i]), "Target_row": int(tgt_row_id[j]),
                "Source_mz": float(src_mz[i]), "Target_mz": float(tgt_mz[j]),
                "Source_RT": float(src_rt[i]), "Target_RT": float(tgt_rt[j]),
                "Predicted_Target_RT": float(prt), "RT_error_min": float(abs(float(tgt_rt[j]) - prt)),
                "Mass_error_ppm": float(ppm_error(src_mz[i], tgt_mz[j])),
                "ACS": acs, "corr_score": corr, "ratio_score": ratio,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out, params
    out = out.sort_values(["Source_row", "ACS", "RT_error_min", "Mass_error_ppm"], ascending=[True, False, True, True]).reset_index(drop=True)
    out["Candidate_rank"] = out.groupby("Source_row").cumcount() + 1
    out["Is_primary_pair"] = out["Candidate_rank"] == 1
    return out, params


def prepare_candidate_pool(src: pd.DataFrame, tgt: pd.DataFrame, src_cols: dict, tgt_cols: dict,
                           settings: RTBridgeSettings) -> pd.DataFrame:
    """Precompute model-independent mass candidates and ACS once per dataset.

    The reference implementation recalculated these identical quantities for
    every model, anchor configuration, and screening workflow. Reuse changes
    only execution time, not the scientific result.
    """
    src = src.reset_index(drop=True)
    tgt = tgt.reset_index(drop=True)
    if src.empty or tgt.empty:
        return pd.DataFrame()

    src_mz = src["__mz"].to_numpy(float)
    src_rt = src["__rt"].to_numpy(float)
    src_row_id = src["__row_id"].to_numpy(int)
    src_qc = src["__qc_mean"].to_numpy(float) if "__qc_mean" in src.columns else np.full(len(src), np.nan)
    tgt_mz = tgt["__mz"].to_numpy(float)
    tgt_rt = tgt["__rt"].to_numpy(float)
    tgt_row_id = tgt["__row_id"].to_numpy(int)
    tgt_qc = tgt["__qc_mean"].to_numpy(float) if "__qc_mean" in tgt.columns else np.full(len(tgt), np.nan)

    target_order = np.argsort(tgt_mz)
    target_mz_sorted = tgt_mz[target_order]
    n_bio = min(len(src_cols.get("bio", [])), len(tgt_cols.get("bio", [])))
    src_bio = src[src_cols.get("bio", [])[:n_bio]].to_numpy(float) if n_bio else np.empty((len(src), 0), dtype=float)
    tgt_bio = tgt[tgt_cols.get("bio", [])[:n_bio]].to_numpy(float) if n_bio else np.empty((len(tgt), 0), dtype=float)

    rows = []
    ppm = float(settings.mass_ppm_candidate)
    for i, mz in enumerate(src_mz):
        target_positions = _mz_window_positions(float(mz), target_mz_sorted, target_order, ppm)
        if target_positions.size == 0:
            continue
        xs = src_bio[i] if n_bio else np.array([], dtype=float)
        for j in target_positions:
            ys = tgt_bio[j] if n_bio else np.array([], dtype=float)
            acs, corr, ratio = _acs_from_arrays(xs, ys, src_qc[i], tgt_qc[j], settings)
            if REQUIRE_POSITIVE_CORRELATION and (not np.isfinite(corr) or corr <= 0):
                continue
            rows.append({
                "__source_pos": int(i),
                "Source_row": int(src_row_id[i]), "Target_row": int(tgt_row_id[j]),
                "Source_mz": float(src_mz[i]), "Target_mz": float(tgt_mz[j]),
                "Source_RT": float(src_rt[i]), "Target_RT": float(tgt_rt[j]),
                "Mass_error_ppm": float(ppm_error(src_mz[i], tgt_mz[j])),
                "ACS": acs, "corr_score": corr, "ratio_score": ratio,
            })
    return pd.DataFrame(rows)


def select_candidate_pairs(src: pd.DataFrame, tgt: pd.DataFrame, src_cols: dict, tgt_cols: dict, anchors: pd.DataFrame,
                           model: str, settings: RTBridgeSettings,
                           prepared_pool: Optional[pd.DataFrame]=None) -> Tuple[pd.DataFrame, dict]:
    """Return exactly the reference candidate pairs with reusable preprocessing."""
    if anchors.empty or len(anchors) < 3:
        return pd.DataFrame(), {"error": "not enough anchors"}

    src = src.reset_index(drop=True)
    pred_rt, params = fit_predict_model(
        model,
        anchors["Source_RT"].values,
        anchors["Target_RT"].values,
        src["__rt"].values,
        settings.random_seed,
    )
    if prepared_pool is None:
        prepared_pool = prepare_candidate_pool(src, tgt, src_cols, tgt_cols, settings)
    if prepared_pool is None or prepared_pool.empty:
        return pd.DataFrame(), params

    source_pos = prepared_pool["__source_pos"].to_numpy(int)
    predicted = np.full(len(prepared_pool), np.nan, dtype=float)
    valid = (source_pos >= 0) & (source_pos < len(pred_rt))
    predicted[valid] = np.asarray(pred_rt, dtype=float)[source_pos[valid]]
    target_rt = prepared_pool["Target_RT"].to_numpy(float)
    rt_error = np.abs(target_rt - predicted)
    keep = np.isfinite(predicted) & np.isfinite(rt_error) & (rt_error <= float(settings.rt_window_min))
    if not np.any(keep):
        return pd.DataFrame(), params

    out = prepared_pool.loc[keep].copy()
    out["Predicted_Target_RT"] = predicted[keep]
    out["RT_error_min"] = rt_error[keep]
    out = out.drop(columns=["__source_pos"], errors="ignore")
    # Restore the exact reference column order before ranking/export.
    out = out[[
        "Source_row", "Target_row", "Source_mz", "Target_mz", "Source_RT", "Target_RT",
        "Predicted_Target_RT", "RT_error_min", "Mass_error_ppm", "ACS", "corr_score", "ratio_score"
    ]]
    # Keep the reference ranking keys and sort settings unchanged.
    out = out.sort_values(
        ["Source_row", "ACS", "RT_error_min", "Mass_error_ppm"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    out["Candidate_rank"] = out.groupby("Source_row").cumcount() + 1
    out["Is_primary_pair"] = out["Candidate_rank"] == 1
    return out, params

def select_primary_pairs(src: pd.DataFrame, tgt: pd.DataFrame, src_cols: dict, tgt_cols: dict, anchors: pd.DataFrame,
                         model: str, settings: RTBridgeSettings) -> Tuple[pd.DataFrame, dict]:
    candidates, params = select_candidate_pairs(src, tgt, src_cols, tgt_cols, anchors, model, settings)
    if candidates.empty:
        return candidates, params
    primary = candidates[candidates["Is_primary_pair"]].copy().reset_index(drop=True)
    return primary, params


def _clean_col_name(name) -> str:
    """Normalize a column name for robust role detection."""
    import re
    s = str(name).strip().lower()
    # Preserve useful adduct information before removing punctuation.
    s = s.replace("[m-h]-", " mminus h ").replace("[m+h]+", " mplus h ")
    s = s.replace("m-h", " mminus h ").replace("m+h", " mplus h ")
    s = s.replace("m/z", " mz ").replace("m.z", " mz ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def _numeric_series(df: pd.DataFrame, col) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")

def _valid_fraction(x: pd.Series, value_range=None) -> float:
    y = pd.to_numeric(x, errors="coerce")
    if value_range is not None:
        lo, hi = value_range
        y = y[(y >= lo) & (y <= hi)]
    return float(y.notna().sum()) / max(len(x), 1)

def _looks_like_rt_col(name: str) -> bool:
    n = _clean_col_name(name)
    return ("rt" in n.split() or n.startswith("rt") or "retention" in n or "time" in n) and not _looks_like_mz_col(name)

def _looks_like_mz_col(name: str) -> bool:
    n = _clean_col_name(name)
    tokens = set(n.split())
    joined = n.replace(" ", "")
    mz_terms = {
        "mz", "mass", "exactmass", "precursor", "ionmz", "measuredmz", "referencemz", "refmz",
        "isotopemz", "adduct", "mminus", "mplus", "mh", "m", "molecularweight"
    }
    return bool(tokens & mz_terms) or any(term in joined for term in ["mz", "exactmass", "precursor", "referencemz", "isotopemz", "mminus", "mplus"])

def _score_standard_col(col, role: str) -> int:
    """Score a standard-file column for roles: source_mz, target_mz, mz, rt5, rt25, name."""
    n = _clean_col_name(col)
    joined = n.replace(" ", "")
    tokens = set(n.split())
    score = 0
    if role == "name":
        if any(t in tokens for t in ["name", "compound", "metabolite", "matched", "standard", "id"]):
            score += 10
        if any(t in tokens for t in ["mz", "mass", "rt", "time", "area", "abundance"]):
            score -= 10
        return score
    if role in {"source_mz", "target_mz", "mz"}:
        if _looks_like_mz_col(col): score += 10
        if "isotope" in tokens or "labeled" in tokens or "labelled" in tokens or "isotopemz" in joined or "isotopelabeledmz" in joined or "isotopelabelledmz" in joined:
            score += 10
        if "reference" in tokens or "ref" in tokens or joined.startswith("ref"): score += 4
        if "mminus" in tokens or "mplus" in tokens or joined in {"mh", "m"}: score += 6
        if role == "source_mz" and any(k in joined for k in ["5min", "rt5", "source", "short"]): score += 6
        if role == "target_mz" and any(k in joined for k in ["25min", "rt25", "target", "long"]): score += 6
        if _looks_like_rt_col(col): score -= 20
        return score
    if role == "rt5":
        if _looks_like_rt_col(col): score += 10
        if any(k in joined for k in ["rt5", "5min", "source", "short"]): score += 8
        if any(k in joined for k in ["rt25", "25min", "target", "long"]): score -= 8
        return score
    if role == "rt25":
        if _looks_like_rt_col(col): score += 10
        if any(k in joined for k in ["rt25", "25min", "target", "long"]): score += 8
        if any(k in joined for k in ["rt5", "5min", "source", "short"]): score -= 8
        return score
    return 0

def _find_standard_name_col(df: pd.DataFrame) -> Optional[str]:
    # Prefer obvious text name columns; fallback to the first mostly nonnumeric column.
    candidates = []
    for c in df.columns:
        score = _score_standard_col(c, "name")
        nonnum = pd.to_numeric(df[c], errors="coerce").isna().mean()
        if score > 0 and nonnum > 0.5:
            candidates.append((score, c))
    if candidates:
        return sorted(candidates, reverse=True)[0][1]
    for c in df.columns:
        if pd.to_numeric(df[c], errors="coerce").isna().mean() > 0.7:
            return c
    return None

def _find_best_numeric_col(df: pd.DataFrame, role: str, value_range=None, exclude: Optional[set]=None) -> Optional[str]:
    exclude = exclude or set()
    best = None
    for c in df.columns:
        if c in exclude:
            continue
        x = _numeric_series(df, c)
        frac = _valid_fraction(x, value_range=value_range)
        if frac < 0.5:
            continue
        score = _score_standard_col(c, role)
        # If the user gives a simple 4-column authentic-standard file such as
        # Name, M-H, RT_5min, RT_25min, the m/z column may not literally contain "mz".
        # Accept numeric columns in the m/z range that are not RT-like.
        if role in {"source_mz", "target_mz", "mz"} and value_range == (40, 2000) and not _looks_like_rt_col(c):
            med = float(np.nanmedian(x))
            if 40 <= med <= 2000:
                score += 2
        if role in {"rt5", "rt25"} and _looks_like_rt_col(c):
            score += 4
        if best is None or score > best[0]:
            best = (score, c)
    if best and best[0] > 0:
        return best[1]
    return None

def _standard_parse_info(path: Path, raw: pd.DataFrame, parsed: pd.DataFrame, detected: dict) -> dict:
    return {
        "File": Path(path).name,
        "Rows_in_file": int(len(raw)),
        "Rows_parsed": int(len(parsed)),
        "Detected_name_col": detected.get("name") or "",
        "Detected_source_mz_col": detected.get("source_mz") or "",
        "Detected_target_mz_col": detected.get("target_mz") or "",
        "Detected_reference_mz_col": detected.get("reference_mz") or "",
        "Detected_RT5_col": detected.get("rt5") or "",
        "Detected_RT25_col": detected.get("rt25") or "",
        "Available_columns": "; ".join(map(str, raw.columns)),
    }

def read_standard_file(path: Optional[Path]) -> pd.DataFrame:
    """Read authentic/isotope standard files with robust automatic column detection.

    Supported examples include:
      - Name, M-H, RT_5min, RT_25min
      - Name, M/Z, RT_5min, RT_25min
      - Compound, isotope m/z, reference m/z, RT5, RT25
      - source_mz/target_mz or 5min_mz/25min_mz style files

    The parser tries to infer the standard name, one or two m/z columns, and RT5/RT25
    columns from the header names and numeric ranges. The detected columns are exported
    in results/standard_file_parse_report.csv.
    """
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    raw = read_table_auto(Path(path))
    if raw.empty:
        return raw
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    detected = {}
    detected["name"] = _find_standard_name_col(df)
    detected["rt5"] = _find_best_numeric_col(df, "rt5", value_range=(0, 60))
    detected["rt25"] = _find_best_numeric_col(df, "rt25", value_range=(0, 60), exclude={detected.get("rt5")})

    # Detect method-specific m/z if present; otherwise use one common m/z/reference column.
    exclude_rt = {c for c in [detected.get("rt5"), detected.get("rt25")] if c}
    detected["source_mz"] = _find_best_numeric_col(df, "source_mz", value_range=(40, 2000), exclude=exclude_rt)
    detected["target_mz"] = _find_best_numeric_col(df, "target_mz", value_range=(40, 2000), exclude=exclude_rt | {detected.get("source_mz")})
    detected["reference_mz"] = _find_best_numeric_col(df, "mz", value_range=(40, 2000), exclude=exclude_rt)

    # Stable-isotope files usually provide one measured isotope m/z plus a separate
    # unlabeled/reference m/z. Verification of the labeled feature must use the
    # isotope m/z for both gradients; the unlabeled reference column is descriptive.
    isotope_mz_cols = [
        c for c in df.columns
        if _looks_like_mz_col(c)
        and "isotope" in _clean_col_name(c).replace(" ", "")
        and "reference" not in _clean_col_name(c).split()
    ]
    if isotope_mz_cols:
        isotope_col = isotope_mz_cols[0]
        detected["source_mz"] = isotope_col
        detected["target_mz"] = isotope_col
        detected["reference_mz"] = isotope_col

    # If there is only one usable m/z column, use it as both source and target reference.
    mz_cols = [detected.get("source_mz"), detected.get("target_mz"), detected.get("reference_mz")]
    mz_cols = [c for c in mz_cols if c]
    if mz_cols:
        common_mz = detected.get("reference_mz") or mz_cols[0]
        detected["source_mz"] = detected.get("source_mz") or common_mz
        detected["target_mz"] = detected.get("target_mz") or common_mz
        detected["reference_mz"] = detected.get("reference_mz") or common_mz

    out = pd.DataFrame()
    if detected.get("name"):
        out["Standard_name"] = df[detected["name"]].astype(str)
    else:
        out["Standard_name"] = [f"standard_{i+1}" for i in range(len(df))]
    if detected.get("source_mz"):
        out["Source_ref_mz"] = pd.to_numeric(df[detected["source_mz"]], errors="coerce")
    if detected.get("target_mz"):
        out["Target_ref_mz"] = pd.to_numeric(df[detected["target_mz"]], errors="coerce")
    if detected.get("reference_mz"):
        out["Reference_mz"] = pd.to_numeric(df[detected["reference_mz"]], errors="coerce")
    if detected.get("rt5"):
        out["RT5_ref"] = pd.to_numeric(df[detected["rt5"]], errors="coerce")
    if detected.get("rt25"):
        out["RT25_ref"] = pd.to_numeric(df[detected["rt25"]], errors="coerce")

    mz_out_cols = [c for c in ["Source_ref_mz", "Target_ref_mz", "Reference_mz"] if c in out.columns]
    if not mz_out_cols:
        empty = pd.DataFrame()
        empty.attrs["parse_info"] = _standard_parse_info(Path(path), raw, empty, detected)
        return empty
    out = out.dropna(subset=mz_out_cols, how="all").reset_index(drop=True)
    out.attrs["parse_info"] = _standard_parse_info(Path(path), raw, out, detected)
    return out

def prepare_standard_index(standards: pd.DataFrame) -> Optional[dict]:
    """Prepare a reusable exact-mass index for one parsed standards table."""
    if standards is None or standards.empty:
        return None
    std = standards.reset_index(drop=True).copy()
    if "Target_ref_mz" in std.columns:
        target_ref = pd.to_numeric(std["Target_ref_mz"], errors="coerce")
    else:
        target_ref = pd.Series(np.nan, index=std.index, dtype=float)
    if "Reference_mz" in std.columns:
        common_ref = pd.to_numeric(std["Reference_mz"], errors="coerce")
        target_ref = target_ref.where(np.isfinite(target_ref), common_ref)
    refs = target_ref.to_numpy(float)
    valid_positions = np.flatnonzero(np.isfinite(refs) & (refs > 0))
    if valid_positions.size == 0:
        return None
    local_order = np.argsort(refs[valid_positions])
    sorted_positions = valid_positions[local_order]
    return {
        "standards": std,
        "target_refs": refs,
        "sorted_refs": refs[sorted_positions],
        "sorted_positions": sorted_positions,
        "source_refs": pd.to_numeric(std["Source_ref_mz"], errors="coerce").to_numpy(float) if "Source_ref_mz" in std.columns else np.full(len(std), np.nan),
        "rt5_refs": pd.to_numeric(std["RT5_ref"], errors="coerce").to_numpy(float) if "RT5_ref" in std.columns else np.full(len(std), np.nan),
        "rt25_refs": pd.to_numeric(std["RT25_ref"], errors="coerce").to_numpy(float) if "RT25_ref" in std.columns else np.full(len(std), np.nan),
        "names": std["Standard_name"].astype(str).to_numpy() if "Standard_name" in std.columns else np.full(len(std), "", dtype=object),
    }


def _reference_denominator_window_positions(observed_mz: float, sorted_reference_mz: np.ndarray,
                                            sorted_positions: np.ndarray, ppm: float) -> np.ndarray:
    """Find references satisfying |observed-reference|/reference <= ppm.

    This preserves the exact denominator used by the original verification code.
    """
    if not np.isfinite(observed_mz) or observed_mz <= 0:
        return np.array([], dtype=int)
    delta = float(ppm) / 1e6
    if delta >= 1:
        return sorted_positions.copy()
    lo = observed_mz / (1.0 + delta)
    hi = observed_mz / (1.0 - delta)
    left = np.searchsorted(sorted_reference_mz, lo, side="left")
    right = np.searchsorted(sorted_reference_mz, hi, side="right")
    return sorted_positions[left:right]


def verify_pairs(pairs: pd.DataFrame, standards: pd.DataFrame, settings: RTBridgeSettings, standard_type: str,
                 standard_index: Optional[dict]=None) -> pd.DataFrame:
    """Verify pairs with the publication workflow rules, using an index."""
    if pairs.empty or standards.empty:
        return pd.DataFrame()
    idx = standard_index or prepare_standard_index(standards)
    if not idx:
        return pd.DataFrame()

    refs = idx["target_refs"]
    sorted_refs = idx["sorted_refs"]
    sorted_positions = idx["sorted_positions"]
    source_refs = idx["source_refs"]
    rt5_refs = idx["rt5_refs"]
    rt25_refs = idx["rt25_refs"]
    names = idx["names"]

    rows = []
    for pr in pairs.itertuples(index=False):
        target_mz = float(getattr(pr, "Target_mz"))
        possible = _reference_denominator_window_positions(
            target_mz, sorted_refs, sorted_positions, settings.verification_mz_ppm
        )
        # The reference implementation traversed standards in original row order.
        possible = np.sort(possible)
        if possible.size == 0:
            continue
        source_mz = float(getattr(pr, "Source_mz"))
        source_rt = float(getattr(pr, "Source_RT"))
        target_rt = float(getattr(pr, "Target_RT"))
        for si in possible:
            target_mz_err = float(ppm_error(target_mz, refs[si]))
            if target_mz_err > settings.verification_mz_ppm:
                continue
            source_ref = source_refs[si]
            source_mz_err = np.nan
            if np.isfinite(source_ref):
                source_mz_err = float(ppm_error(source_mz, source_ref))
                if source_mz_err > settings.verification_mz_ppm:
                    continue

            rt5_ref = rt5_refs[si]
            rt25_ref = rt25_refs[si]
            rt5_err = abs(source_rt - rt5_ref) if np.isfinite(rt5_ref) else np.nan
            rt25_err = abs(target_rt - rt25_ref) if np.isfinite(rt25_ref) else np.nan
            if np.isfinite(rt5_err) and rt5_err > settings.verification_rt5_min:
                continue
            if np.isfinite(rt25_err) and rt25_err > settings.verification_rt25_min:
                continue
            rows.append({
                "Standard_type": standard_type,
                "Standard_name": names[si],
                "Source_row": getattr(pr, "Source_row"), "Target_row": getattr(pr, "Target_row"),
                "Source_ref_mz": source_ref if np.isfinite(source_ref) else np.nan,
                "Target_ref_mz": refs[si],
                "Source_mz": source_mz, "Target_mz": target_mz,
                "Source_mz_error_ppm": source_mz_err if np.isfinite(source_mz_err) else np.nan,
                "Target_mz_error_ppm": target_mz_err,
                "RT5_error_min": float(rt5_err) if np.isfinite(rt5_err) else np.nan,
                "RT25_error_min": float(rt25_err) if np.isfinite(rt25_err) else np.nan,
            })
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def annotate_verification_on_pairs(pair_df: pd.DataFrame, verification_df: pd.DataFrame) -> pd.DataFrame:
    """Add verification-hit columns to primary/candidate pair tables.

    The annotation is based on Dataset, Workflow, Model, anchor configuration,
    Source_row, and Target_row so users can see which selected feature pairs are
    verified by authentic standards or isotope-labeled standards.
    """
    if pair_df is None or pair_df.empty:
        return pair_df
    out = pair_df.copy()
    out["Verified_any"] = False
    out["Authentic_verified"] = False
    out["Isotope_verified"] = False
    out["Verified_metabolite_names"] = ""
    out["Authentic_standard_names"] = ""
    out["Isotope_standard_names"] = ""
    out["Verification_types"] = ""
    if verification_df is None or verification_df.empty:
        return out
    key_cols = ["Dataset", "Anchor_configuration", "Workflow", "Model", "Source_row", "Target_row"]
    missing_pairs = [c for c in key_cols if c not in out.columns]
    missing_ver = [c for c in key_cols if c not in verification_df.columns]
    if missing_pairs or missing_ver:
        return out

    def _join_unique(x):
        vals = [str(v) for v in x.dropna().astype(str).tolist() if str(v).strip() and str(v).lower() != "nan"]
        vals = list(dict.fromkeys(vals))
        return "; ".join(vals)

    # General summary by pair key
    g = verification_df.groupby(key_cols, dropna=False).agg(
        Verified_metabolite_names=("Standard_name", _join_unique),
        Verification_types=("Standard_type", _join_unique),
    ).reset_index()
    out = out.drop(columns=["Verified_metabolite_names", "Verification_types"], errors="ignore").merge(g, on=key_cols, how="left")

    # Authentic and isotope-specific names
    for stype, flag_col, name_col in [
        ("Authentic", "Authentic_verified", "Authentic_standard_names"),
        ("Isotope", "Isotope_verified", "Isotope_standard_names"),
    ]:
        sub = verification_df[verification_df["Standard_type"].astype(str).str.lower().str.contains(stype.lower(), na=False)]
        if sub.empty:
            continue
        gg = sub.groupby(key_cols, dropna=False).agg(**{name_col: ("Standard_name", _join_unique)}).reset_index()
        out = out.drop(columns=[name_col], errors="ignore").merge(gg, on=key_cols, how="left")
        out[flag_col] = out.get(name_col, "").fillna("").astype(str).str.len() > 0

    for c in ["Verified_metabolite_names", "Authentic_standard_names", "Isotope_standard_names", "Verification_types"]:
        if c in out.columns:
            out[c] = out[c].fillna("")
    out["Verified_any"] = out["Verified_metabolite_names"].fillna("").astype(str).str.len() > 0
    return out

# --------------------------- main workflow ---------------------------
def run_rtbridge(dataset_specs: List[DatasetSpec], workdir: Path, output_dir: Path, settings: RTBridgeSettings,
                 authentic_files: Optional[Dict[str, str]]=None, isotope_files: Optional[Dict[str, str]]=None,
                 progress=None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    authentic_files = authentic_files or {}; isotope_files = isotope_files or {}
    models = [m for m in (settings.selected_models or DEFAULT_MODEL_NAMES) if m in EXPANDED_MODEL_NAMES]
    if not models:
        models = DEFAULT_MODEL_NAMES.copy()
    all_results, all_anchors, all_pairs, all_candidate_pairs, all_ver, all_audit, all_diag = [], [], [], [], [], [], []
    feature_summary, data_cache, anchor_cache, donor_report = [], {}, {}, []
    standard_cache_auth, standard_cache_iso = {}, {}
    standard_index_auth, standard_index_iso = {}, {}

    for spec in dataset_specs:
        if progress: progress(f"Reading {spec.dataset}")
        src, src_cols = read_feature_table(
            workdir / spec.source_file, settings, spec.dataset + "_source",
            expected_replicates=spec.technical_replicates,
            expected_samples=spec.expected_biological_samples,
        )
        tgt, tgt_cols = read_feature_table(
            workdir / spec.target_file, settings, spec.dataset + "_target",
            expected_replicates=spec.technical_replicates,
            expected_samples=spec.expected_biological_samples,
        )
        if src_cols.get("n_donors") != tgt_cols.get("n_donors"):
            raise ColumnError(
                f"{spec.dataset}: source has {src_cols.get('n_biological_samples')} biological-sample summaries but "
                f"target has {tgt_cols.get('n_biological_samples')}."
            )
        source_ids = [_canonical_biological_sample_id(x) for x in src_cols.get("donor_ids", [])]
        target_ids = [_canonical_biological_sample_id(x) for x in tgt_cols.get("donor_ids", [])]
        if source_ids and target_ids and source_ids != target_ids:
            raise ColumnError(
                f"{spec.dataset}: source and target biological-sample groups do not align. "
                f"Source groups: {src_cols.get('biological_sample_ids')}; "
                f"target groups: {tgt_cols.get('biological_sample_ids')}."
            )
        for role, cols_now in [("source", src_cols), ("target", tgt_cols)]:
            for group in cols_now.get("biological_sample_groups", []):
                donor_report.append({
                    "Dataset": spec.dataset,
                    "Sample_type": spec.matrix,
                    "Ionization_mode": spec.mode,
                    "Table_role": role,
                    "Biological_sample_index": group["sample_index"],
                    "Biological_sample_ID": group["sample_id"],
                    "Canonical_biological_sample_ID": _canonical_biological_sample_id(group["sample_id"]),
                    "Grouping_method": group["grouping_method"],
                    "Summary_method": group["summary_method"],
                    "Summary_column": group["summary_column"],
                    "Raw_injection_columns": " | ".join(group["raw_injection_columns"]),
                    "Number_of_injections": group["number_of_injections"],
                    "Detected_QC_columns": " | ".join(cols_now.get("qc", [])),
                    "Number_of_QC_injections": len(cols_now.get("qc", [])),
                    "Excluded_numeric_columns": " | ".join(cols_now.get("excluded_numeric", [])),
                })
        data_cache[spec.dataset] = (src, tgt, src_cols, tgt_cols, spec)
        anchors = build_anchors(src, tgt, src_cols, tgt_cols, settings)
        anchors["Dataset"] = spec.dataset; anchors["Matrix"] = spec.matrix; anchors["Mode"] = spec.mode; anchors["Anchor_configuration"] = spec.mode
        anchor_cache[(spec.dataset, spec.mode)] = anchors
        feature_summary.append({
            "Dataset": spec.dataset, "Matrix": spec.matrix, "Mode": spec.mode,
            "Source_features_after_QC": len(src), "Target_features_after_QC": len(tgt),
            "Unique_anchor_pairs": len(anchors),
            "High_RT5_anchors_GE_2_5": int((anchors["Source_RT"] >= 2.5).sum()) if len(anchors) else 0,
            "High_RT25_anchors_GE_7": int((anchors["Target_RT"] >= 7.0).sum()) if len(anchors) else 0,
            "Median_anchor_ACS": float(anchors["ACS"].median()) if len(anchors) else np.nan,
            "Independent_biological_samples": int(src_cols.get("n_biological_samples", 0)),
            "Technical_injections_per_biological_sample": src_cols.get("replicate_count_summary", ""),
            "Replicate_grouping_method": src_cols.get("replicate_grouping_method", ""),
            "Biological_summary_method": src_cols.get("biological_summary", ""),
            "Source_QC_injections": len(src_cols.get("qc", [])),
            "Target_QC_injections": len(tgt_cols.get("qc", [])),
        })

    for matrix in sorted(set(s.matrix for s in dataset_specs)):
        parts = [a for (ds, mode), a in anchor_cache.items() if data_cache[ds][4].matrix == matrix]
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            combined["Anchor_configuration"] = "COMBINED"
            for spec in [s for s in dataset_specs if s.matrix == matrix]:
                anchor_cache[(spec.dataset, "COMBINED")] = combined.copy()

    standard_report_rows = []
    for mode, fname in authentic_files.items():
        df_std = read_standard_file(workdir / fname) if fname else pd.DataFrame()
        standard_cache_auth[mode] = df_std
        standard_index_auth[mode] = prepare_standard_index(df_std)
        info = dict(df_std.attrs.get("parse_info", {}))
        if info:
            info.update({"Mode": mode, "Standard_type": "Authentic"})
            standard_report_rows.append(info)
    for mode, fname in isotope_files.items():
        df_std = read_standard_file(workdir / fname) if fname else pd.DataFrame()
        standard_cache_iso[mode] = df_std
        standard_index_iso[mode] = prepare_standard_index(df_std)
        info = dict(df_std.attrs.get("parse_info", {}))
        if info:
            info.update({"Mode": mode, "Standard_type": "Isotope"})
            standard_report_rows.append(info)

    # Save parsed standard files for paper-ready supplementary tables.
    rd_pre = output_dir / "results"
    rd_pre.mkdir(parents=True, exist_ok=True)
    parsed_auth_parts = []
    for mode, df_std in standard_cache_auth.items():
        if df_std is not None and len(df_std):
            parsed_auth_parts.append(df_std.assign(Mode=mode, Standard_type="Authentic"))
    if parsed_auth_parts:
        pd.concat(parsed_auth_parts, ignore_index=True).to_csv(rd_pre / "authentic_standards_parsed_all.csv", index=False)
    parsed_iso_parts = []
    for mode, df_std in standard_cache_iso.items():
        if df_std is not None and len(df_std):
            parsed_iso_parts.append(df_std.assign(Mode=mode, Standard_type="Isotope"))
    if parsed_iso_parts:
        pd.concat(parsed_iso_parts, ignore_index=True).to_csv(rd_pre / "isotope_standards_parsed_all.csv", index=False)

    for spec in dataset_specs:
        src, tgt, src_cols, tgt_cols, _ = data_cache[spec.dataset]
        if len(models) > 1:
            if progress: progress(f"Preparing reusable candidate pool for {spec.dataset}")
            prepared_pool_for_dataset = prepare_candidate_pool(src, tgt, src_cols, tgt_cols, settings)
        else:
            prepared_pool_for_dataset = None
        for config in [spec.mode, "COMBINED"]:
            anchors = anchor_cache.get((spec.dataset, config), pd.DataFrame())
            if anchors.empty:
                continue
            workflows = {"All selected anchors": anchors}
            # Export all selected anchors and the GAM-screened anchors so figures
            # can compare the same selected anchors before and after screening.
            all_anchors.append(anchors.assign(Dataset=spec.dataset, Matrix=spec.matrix, Mode=spec.mode, Anchor_configuration=config, Workflow="All selected anchors"))
            screened, audit = gam_based_screen(anchors, settings)
            workflows["GAM-screened anchors"] = screened
            if not audit.empty:
                audit["Dataset"] = spec.dataset; audit["Matrix"] = spec.matrix; audit["Mode"] = spec.mode; audit["Anchor_configuration"] = config
                all_audit.append(audit)
            all_anchors.append(screened.assign(Dataset=spec.dataset, Matrix=spec.matrix, Mode=spec.mode, Anchor_configuration=config, Workflow="GAM-screened anchors"))
            for workflow, anchor_set in workflows.items():
                for model in models:
                    if progress: progress(f"{spec.dataset} | {config} | {workflow} | {model}")
                    prepared_pool = prepared_pool_for_dataset
                    if prepared_pool is None and len(models) == 1:
                        candidates, params = _select_candidate_pairs_direct(src, tgt, src_cols, tgt_cols, anchor_set, model, settings)
                    else:
                        candidates, params = select_candidate_pairs(src, tgt, src_cols, tgt_cols, anchor_set, model, settings, prepared_pool=prepared_pool)
                    candidates = candidates.assign(Dataset=spec.dataset, Matrix=spec.matrix, Mode=spec.mode, Anchor_configuration=config, Workflow=workflow, Model=model)
                    pairs = candidates[candidates.get("Is_primary_pair", False)].copy().reset_index(drop=True) if len(candidates) else pd.DataFrame()
                    v1 = verify_pairs(pairs, standard_cache_auth.get(spec.mode, pd.DataFrame()), settings, "Authentic", standard_index_auth.get(spec.mode))
                    v2 = verify_pairs(pairs, standard_cache_iso.get(spec.mode, pd.DataFrame()), settings, "Isotope", standard_index_iso.get(spec.mode))
                    if len(v1) or len(v2):
                        ver = pd.concat([v1, v2], ignore_index=True).assign(Dataset=spec.dataset, Matrix=spec.matrix, Mode=spec.mode, Anchor_configuration=config, Workflow=workflow, Model=model)
                        all_ver.append(ver)
                    if getattr(settings, "export_all_candidate_pairs", True):
                        all_candidate_pairs.append(candidates)
                    all_pairs.append(pairs)
                    if getattr(settings, "make_model_fit_diagnostics", True):
                        diag = model_anchor_diagnostics(anchor_set, model, workflow, spec.dataset, spec.matrix, spec.mode, config, settings)
                        if not diag.empty:
                            all_diag.append(diag)
                    all_results.append({
                        "Dataset": spec.dataset, "Matrix": spec.matrix, "Mode": spec.mode,
                        "Anchor_configuration": config, "Workflow": workflow, "Model": model,
                        "Model_group": next((g for g, ms in MODEL_GROUPS.items() if model in ms and g != "All available"), "Other"),
                        "Initial_anchors": len(anchors), "Final_anchors": len(anchor_set),
                        "Candidate_pairs": len(candidates),
                        "Source_features_with_candidates": int(candidates["Source_row"].nunique()) if len(candidates) else 0,
                        "Nonprimary_candidate_pairs": int(len(candidates) - len(pairs)),
                        "Primary_pairs": len(pairs),
                        "Median_RT_error_min": float(pairs["RT_error_min"].median()) if len(pairs) else np.nan,
                        "P90_RT_error_min": float(pairs["RT_error_min"].quantile(0.90)) if len(pairs) else np.nan,
                        "Median_ACS": float(pairs["ACS"].median()) if len(pairs) else np.nan,
                        "Authentic_hits": int(len(v1)), "Isotope_hits": int(len(v2)), "Total_verification": int(len(v1) + len(v2)),
                        "Model_parameters": json.dumps(params),
                    })
    results = pd.DataFrame(all_results)
    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    candidate_pairs = pd.concat(all_candidate_pairs, ignore_index=True) if all_candidate_pairs else pd.DataFrame()
    anchors = pd.concat(all_anchors, ignore_index=True) if all_anchors else pd.DataFrame()
    ver = pd.concat(all_ver, ignore_index=True) if all_ver else pd.DataFrame()
    # Add verification annotations directly to the pair tables for easier review.
    pairs = annotate_verification_on_pairs(pairs, ver)
    candidate_pairs = annotate_verification_on_pairs(candidate_pairs, ver)
    audit = pd.concat(all_audit, ignore_index=True) if all_audit else pd.DataFrame()
    diag = pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame()
    feature_summary = pd.DataFrame(feature_summary)
    rd = output_dir / "results"
    results.to_csv(rd / "model_metrics_all.csv", index=False)
    pairs.to_csv(rd / "primary_pairs_all.csv", index=False)
    candidate_pairs.to_csv(rd / "candidate_pairs_all.csv", index=False)
    anchors.to_csv(rd / "anchors_all.csv", index=False)
    ver.to_csv(rd / "verification_hits_all.csv", index=False)
    audit.to_csv(rd / "GAM_based_screen_audit.csv", index=False)
    diag.to_csv(rd / "model_anchor_residual_diagnostics_all.csv", index=False)
    feature_summary.to_csv(rd / "feature_anchor_summary.csv", index=False)
    pd.DataFrame(donor_report).to_csv(rd / "biological_sample_aggregation_report.csv", index=False)
    # Export a parser report for optional authentic/isotope standard files so users can
    # confirm which columns were detected as m/z and RT references.
    if standard_report_rows:
        pd.DataFrame(standard_report_rows).to_csv(rd / "standard_file_parse_report.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "Standard_type", "Mode", "File", "Rows_in_file", "Rows_parsed",
            "Detected_name_col", "Detected_source_mz_col", "Detected_target_mz_col",
            "Detected_reference_mz_col", "Detected_RT5_col", "Detected_RT25_col",
            "Available_columns"
        ]).to_csv(rd / "standard_file_parse_report.csv", index=False)
    with open(output_dir / "run_settings.json", "w", encoding="utf-8") as f:
        d = asdict(settings)
        d["selected_models"] = models
        d["datasets"] = [asdict(spec) for spec in dataset_specs]
        d["biological_summary_behavior"] = (
            "Technical replicate injections are averaged within each independent biological "
            "sample before correlation, ratio error, ratio score, and ACS calculations."
        )
        json.dump(d, f, indent=2)
    return {"results": results, "pairs": pairs, "candidate_pairs": candidate_pairs, "anchors": anchors, "verification": ver, "audit": audit, "diagnostics": diag, "feature_summary": feature_summary, "output_dir": output_dir}

def zip_output(output_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in output_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(output_dir)))
    return zip_path
