from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rtbridge.core import DatasetSpec, RTBridgeSettings, run_rtbridge
from rtbridge.plotting import make_all_figures_tables

DEFAULT_MODELS = [
    "Linear regression",
    "Piecewise linear",
    "LOESS",
    "PCHIP",
    "Smoothing spline/GAM",
    "Monotone GAM",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RTBridge from a dataset manifest.")
    parser.add_argument(
        "manifest",
        type=Path,
        help=(
            "CSV manifest with dataset, sample_type or matrix, mode, source_file, "
            "target_file, and optional technical_replicates and expected_biological_samples columns."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("RTBridge_outputs_cli"), help="Output directory.")
    parser.add_argument("--workdir", type=Path, default=Path.cwd(), help="Base directory for relative paths in the manifest.")
    parser.add_argument("--u", type=float, default=20.0, help="Within-method uniqueness window in ppm.")
    parser.add_argument("--a", type=float, default=10.0, help="Cross-method anchor-matching window in ppm.")
    parser.add_argument("--c", type=float, default=10.0, help="Final candidate-matching window in ppm.")
    parser.add_argument("--corr-weight", type=float, default=0.5, help="ACS correlation-component weight.")
    parser.add_argument("--ratio-weight", type=float, default=0.5, help="ACS ratio-component weight.")
    parser.add_argument("--square-correlation", action="store_true", help="Square positive Pearson r before ACS weighting.")
    parser.add_argument("--auth-pos", default="", help="Authentic-standard file for POS mode.")
    parser.add_argument("--auth-neg", default="", help="Authentic-standard file for NEG mode.")
    parser.add_argument("--iso-pos", default="", help="Isotope-labeled standard file for POS mode.")
    parser.add_argument("--iso-neg", default="", help="Isotope-labeled standard file for NEG mode.")
    return parser.parse_args()


def _optional_positive_int(value) -> int | None:
    if pd.isna(value) or str(value).strip() in {"", "0", "auto", "Auto"}:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"Expected a positive integer or blank/0/auto, received: {value}")
    return parsed


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    df = pd.read_csv(manifest)
    matrix_col = "sample_type" if "sample_type" in df.columns else "matrix"
    required = {"dataset", matrix_col, "mode", "source_file", "target_file"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    specs = []
    for row in df.to_dict(orient="records"):
        specs.append(
            DatasetSpec(
                dataset=str(row["dataset"]),
                matrix=str(row[matrix_col]),
                mode=str(row["mode"]),
                source_file=str(row["source_file"]),
                target_file=str(row["target_file"]),
                technical_replicates=_optional_positive_int(row.get("technical_replicates")),
                expected_biological_samples=_optional_positive_int(row.get("expected_biological_samples")),
            )
        )

    settings = RTBridgeSettings(
        mass_ppm_unique=args.u,
        mass_ppm_anchor=args.a,
        mass_ppm_candidate=args.c,
        correlation_weight=args.corr_weight,
        ratio_weight=args.ratio_weight,
        square_correlation=args.square_correlation,
        selected_models=DEFAULT_MODELS,
    )
    output_dir = args.output.resolve()
    authentic = {k:v for k,v in {"POS":args.auth_pos,"NEG":args.auth_neg}.items() if v}
    isotope = {k:v for k,v in {"POS":args.iso_pos,"NEG":args.iso_neg}.items() if v}
    result = run_rtbridge(specs, args.workdir.resolve(), output_dir, settings, authentic, isotope)
    make_all_figures_tables(output_dir, dpi=settings.output_dpi)
    print(f"Done. Outputs: {output_dir}")
    print(result["results"].head())


if __name__ == "__main__":
    main()
