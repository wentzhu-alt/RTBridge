from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import streamlit as st

from rtbridge.core import (
    DatasetSpec,
    RTBridgeSettings,
    run_rtbridge,
    zip_output,
    EXPANDED_MODEL_NAMES,
    DEFAULT_MODEL_NAMES,
    MODEL_GROUPS,
    MODEL_NOTES,
)
from rtbridge.plotting import format_publication_table, make_all_figures_tables

st.set_page_config(page_title="RTBridge", layout="wide")
st.title("RTBridge: Retention-Time Bridging across Gradients for LC-HRMS Feature Correspondence")
st.caption("Independent biological-sample-level scoring with flexible sample and replicate recognition")

if "rtbridge_last_run" not in st.session_state:
    st.session_state.rtbridge_last_run = None

with st.expander("Input file requirements", expanded=False):
    st.markdown(
        """
Feature tables can be CSV, TXT/TSV, or Excel files with at least:

- an m/z column, named like `mz`, `m/z`, or `mass`
- an RT column, named like `rt` or `retention time`
- numeric abundance columns for biological samples and, ideally, pooled QC samples
- QC columns containing `QC` or `pool` in the column name; QC injections are handled separately and are not counted as biological samples
- technical-injection columns named consistently, for example `Sample01_rep1`, `Sample01_rep2`, ...

RTBridge can automatically recognize any consistent number of technical injections per biological sample. If the names do not contain sample and injection identifiers, provide the expected injection count in the dataset panel so consecutive sample-major columns can be grouped safely. Averaging within each independent biological sample is always applied and cannot be disabled.

Standard files are optional. They may contain one common m/z/reference-m/z column or separate source/target m/z columns. RT5 and RT25 columns are optional but recommended. Windows-encoded files are supported.
        """
    )

st.sidebar.header("1. Analysis settings")
model_preset = st.sidebar.selectbox("Model preset", list(MODEL_GROUPS.keys()), index=0)
default_models = MODEL_GROUPS[model_preset]
selected_models = st.sidebar.multiselect(
    "RT-transfer models to run",
    options=EXPANDED_MODEL_NAMES,
    default=default_models,
    help="The six manuscript models are recommended for routine runs. Large exploratory model sets take longer.",
)
if not selected_models:
    st.sidebar.warning("No model selected. The six manuscript models will be used.")
    selected_models = DEFAULT_MODEL_NAMES.copy()

with st.sidebar.expander("Model notes", expanded=False):
    for model_name in selected_models:
        st.write(f"**{model_name}:** {MODEL_NOTES.get(model_name, '')}")

st.sidebar.header("2. Speed and output")
fast_mode = st.sidebar.checkbox(
    "Fast mode",
    value=False,
    help="Optional. Numerical model results remain unchanged, but the largest candidate export, exploratory diagnostics, and TIFF rendering are skipped.",
)
export_all_candidate_pairs = st.sidebar.checkbox(
    "Export all candidate-pair rows",
    value=not fast_mode,
    disabled=fast_mode,
    help="This file can be large. Primary-pair results and manuscript metrics are still produced when it is disabled.",
)
make_model_fit_diagnostics = st.sidebar.checkbox(
    "Generate every model fit/residual diagnostic",
    value=not fast_mode,
    disabled=fast_mode,
    help="Useful for detailed exploration but slower, especially at 600–900 dpi.",
)

with st.sidebar.expander("About RTBridge", expanded=False):
    st.markdown(
        "Default behavior: Correlation, ratio error, ratio score, and ACS are computed "
        "from independent biological-sample-level summaries obtained by averaging technical "
        "replicate injections for each biological sample. In the present manuscript study, "
        "the biological samples are four donors. RTBridge is not limited to this design and "
        "supports any number of sample types, biological samples, technical injections, and QC injections when grouping is identifiable."
    )

st.sidebar.header("3. Matching thresholds")
mass_ppm_unique = st.sidebar.number_input(
    "Within-method uniqueness window (ppm)", value=20.0, min_value=1.0, max_value=100.0, step=1.0,
    help="Applied separately within each feature table. A larger value defines uniqueness more conservatively and generally reduces the anchor pool.",
)
mass_ppm_anchor = st.sidebar.number_input(
    "Anchor mass window (ppm)", value=10.0, min_value=1.0, max_value=100.0, step=1.0,
    help="Applied across the source and target methods when constructing reciprocal one-to-one high-confidence training anchors. The publication default is 10 ppm as a balanced setting that is neither overly permissive nor overly restrictive.",
)
mass_ppm_candidate = st.sidebar.number_input(
    "Candidate pair mass window (ppm)", value=10.0, min_value=1.0, max_value=100.0, step=1.0,
    help="Applied after RT-model fitting to generate final candidate pairs. It may be wider than the anchor window because final candidates must also pass the fixed projected-RT window, positive biological-sample-level correlation, and ACS ranking. The sensitivity analysis selected 10 ppm.",
)

with st.sidebar.expander("How the three mass windows differ", expanded=False):
    st.markdown(
        """
1. **Uniqueness window:** removes locally ambiguous m/z features within each method.  
2. **Anchor window:** pairs the remaining unique source and target features for model training.  
3. **Candidate window:** searches for final feature correspondences after RT prediction.

The publication defaults are **20/10/10 ppm** (U/A/C): 20 ppm for within-method uniqueness, 10 ppm for cross-method anchor matching, and 10 ppm for final candidate matching. Sensitivity analysis showed very similar performance for neighboring combinations; 20/10/10 was selected as a balanced compromise that is neither too permissive nor too restrictive.
        """
    )

st.sidebar.header("4. ACS correlation penalty and weights")
correlation_penalty = st.sidebar.radio(
    "Correlation penalty",
    ["Linear positive r", "Squared positive r"],
    index=0,
    horizontal=True,
    help=(
        "Linear uses r directly. Squared uses r² and therefore penalizes weak and moderate "
        "positive correlations more strongly. In both cases, undefined, zero, and negative "
        "correlations are deleted before the transformation and before ACS weighting."
    ),
)
square_correlation = correlation_penalty == "Squared positive r"

weight_preset = st.sidebar.selectbox(
    "Starting weight preset",
    ["Equal 0.5/0.5", "Correlation-heavy 0.7/0.3", "Correlation-heavy 0.8/0.2", "Custom"],
    index=0,
    help="Choose a starting point, then edit either weight manually below.",
)
_preset_weights = {
    "Equal 0.5/0.5": (0.5, 0.5),
    "Correlation-heavy 0.7/0.3": (0.7, 0.3),
    "Correlation-heavy 0.8/0.2": (0.8, 0.2),
    "Custom": (0.5, 0.5),
}
_default_corr_weight, _default_ratio_weight = _preset_weights[weight_preset]
correlation_weight = st.sidebar.number_input(
    "Correlation weight",
    value=float(_default_corr_weight),
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    help="Weight applied to positive r or positive r². Weights are normalized to sum to 1.",
)
ratio_weight = st.sidebar.number_input(
    "Ratio-score weight",
    value=float(_default_ratio_weight),
    min_value=0.0,
    max_value=10.0,
    step=0.05,
    help="Weight applied to the QC-normalized proportional-agreement score. Weights are normalized to sum to 1.",
)
_weight_sum = correlation_weight + ratio_weight
if _weight_sum <= 0:
    st.sidebar.error("At least one ACS weight must be greater than zero.")
else:
    st.sidebar.caption(
        f"Normalized ACS weights: correlation = {correlation_weight/_weight_sum:.3f}; "
        f"ratio = {ratio_weight/_weight_sum:.3f}."
    )
st.sidebar.info(
    "Mandatory order: calculate Pearson r across independent biological-sample means; "
    "delete pairs with undefined, zero, or negative r; then apply the selected linear or squared "
    "positive-correlation penalty; finally combine it with the ratio score."
)
st.sidebar.caption(
    "The selected penalty and weights are applied end-to-end to anchor filtering and final candidate ranking, "
    "so they can change anchors, RT models, candidate pairs, primary links, and verification results."
)

if mass_ppm_unique < mass_ppm_anchor:
    st.sidebar.info("The uniqueness window is narrower than the anchor window. Reciprocal matching still removes ambiguity, but a uniqueness window at least as wide as the anchor window is usually easier to interpret.")

settings = RTBridgeSettings(
    mass_ppm_unique=mass_ppm_unique,
    mass_ppm_anchor=mass_ppm_anchor,
    mass_ppm_candidate=mass_ppm_candidate,
    qc_mean_min=st.sidebar.number_input("Minimum QC mean", value=100.0, min_value=0.0),
    qc_cv_max=st.sidebar.number_input("Maximum QC CV", value=0.30, min_value=0.0, max_value=2.0, step=0.05),
    acs_min=st.sidebar.number_input("Minimum anchor ACS", value=0.50, min_value=0.0, max_value=1.0, step=0.05),
    correlation_weight=correlation_weight,
    ratio_weight=ratio_weight,
    square_correlation=square_correlation,
    rt_window_min=st.sidebar.number_input("Predicted RT window for candidates (min)", value=0.35, min_value=0.01, max_value=5.0, step=0.05),
    verification_mz_ppm=st.sidebar.number_input("Verification m/z window (ppm)", value=10.0, min_value=1.0, max_value=100.0, step=1.0),
    verification_rt5_min=st.sidebar.number_input("Verification RT5 window (min)", value=0.20, min_value=0.01, max_value=5.0, step=0.01),
    verification_rt25_min=st.sidebar.number_input("Verification RT25 window (min)", value=0.20, min_value=0.01, max_value=5.0, step=0.01),
    gam_screen_iterations=st.sidebar.number_input(
        "GAM-based residual-screening iterations",
        value=2,
        min_value=0,
        max_value=5,
        step=1,
        help=(
            "All selected anchors include every anchor passing the mass, QC, correlation, "
            "and abundance-concordance criteria. GAM-screened anchors are the subset "
            "remaining after iterative removal of high-residual anchors. Both are "
            "evaluated with every selected RT-transfer model."
        ),
    ),
    residual_multiplier=st.sidebar.number_input("Residual cutoff multiplier", value=2.0, min_value=0.5, max_value=10.0, step=0.5),
    output_dpi=st.sidebar.selectbox("Figure resolution", [300, 600, 900], index=0 if fast_mode else 1),
    selected_models=selected_models,
    export_all_candidate_pairs=export_all_candidate_pairs and not fast_mode,
    make_model_fit_diagnostics=make_model_fit_diagnostics and not fast_mode,
)

st.sidebar.caption(
    "Anchor-set comparison: All selected anchors (no residual-based removal) "
    "versus GAM-screened anchors (GAM-based sensitivity analysis)."
)

st.header("5. Upload sample types and feature tables")
st.write(
    "Add one dataset for each sample-type × ionization-mode combination. "
    "The current-study preset contains plasma and whole blood in positive and negative modes (four datasets), "
    "but additional sample types and replicate designs can be added."
)
st.info(
    "Current-study preset: two sample types (plasma and whole blood), four independent biological samples per dataset, "
    "one pooled-QC sample planned for three injections, and three technical injections per biological sample. After preprocessing, the supplied tables contain two or three usable QC columns, all of which are used. "
    "These are starting values only; RTBridge can analyze other sample types and other replicate counts."
)
n_datasets = st.number_input(
    "Number of datasets / sample-type × mode combinations",
    min_value=1, max_value=60, value=4, step=1,
)

SAMPLE_TYPE_OPTIONS = [
    "Plasma",
    "Whole blood",
    "Serum",
    "Urine",
    "Cerebrospinal fluid",
    "Saliva",
    "Stool / feces",
    "Tissue",
    "Cell pellet",
    "Cell culture medium",
    "Bile",
    "Breath condensate",
    "Dried blood spot",
    "Other / custom",
]
REPLICATE_OPTIONS = ["Auto-detect from column names"] + list(range(1, 51))


def default_dataset(index: int):
    defaults = [
        ("PL_NEG", "Plasma", "NEG"),
        ("PL_POS", "Plasma", "POS"),
        ("WB_NEG", "Whole blood", "NEG"),
        ("WB_POS", "Whole blood", "POS"),
    ]
    return defaults[index] if index < len(defaults) else (f"Dataset_{index + 1}", "Other / custom", "NEG")


uploaded_files = {}
specs = []
partial_uploads = []
for i in range(n_datasets):
    dataset0, sample_type0, mode0 = default_dataset(i)
    st.subheader(f"Dataset {i + 1}")
    c1, c2, c3 = st.columns(3)
    dataset = c1.text_input(f"Dataset name {i + 1}", value=dataset0, key=f"dataset_{i}").strip()
    sample_index = SAMPLE_TYPE_OPTIONS.index(sample_type0) if sample_type0 in SAMPLE_TYPE_OPTIONS else len(SAMPLE_TYPE_OPTIONS) - 1
    sample_type_choice = c2.selectbox(
        f"Sample type {i + 1}", SAMPLE_TYPE_OPTIONS, index=sample_index, key=f"sample_type_{i}"
    )
    mode_choice = c3.selectbox(
        f"Ionization mode {i + 1}", ["NEG", "POS", "Other / custom"],
        index=0 if mode0 == "NEG" else 1, key=f"mode_choice_{i}"
    )

    c4, c5, c6 = st.columns(3)
    if sample_type_choice == "Other / custom":
        sample_type = c4.text_input(
            f"Custom sample type {i + 1}", value="Custom sample", key=f"custom_sample_type_{i}"
        ).strip()
    else:
        sample_type = sample_type_choice
        c4.caption(f"Selected sample type: {sample_type}")

    if mode_choice == "Other / custom":
        mode = c5.text_input(f"Custom mode {i + 1}", value="MODE", key=f"custom_mode_{i}").strip().upper()
    else:
        mode = mode_choice
        c5.caption(f"Selected mode: {mode}")

    replicate_default_index = REPLICATE_OPTIONS.index(3) if i < 4 else 0
    replicate_choice = c6.selectbox(
        f"Technical injections per biological sample {i + 1}",
        REPLICATE_OPTIONS,
        index=replicate_default_index,
        key=f"technical_replicates_{i}",
        help=(
            "Averaging is always applied. Choose Auto when column names contain sample and "
            "replicate identifiers. Choose a number, such as 3 or 5, only when columns are arranged "
            "as consecutive injections for each biological sample. This setting assists grouping only; "
            "technical injections are always averaged within each biological sample."
        ),
    )
    technical_replicates = None if isinstance(replicate_choice, str) else int(replicate_choice)

    expected_samples_default = 4 if i < 4 else 0
    expected_samples_value = st.number_input(
        f"Expected independent biological samples for {dataset} (0 = auto)",
        min_value=0, max_value=1000, value=expected_samples_default, step=1, key=f"expected_samples_{i}",
        help=(
            "Optional validation only. The current study uses 4 independent biological samples per dataset. "
            "QC injections are recognized separately and are not included in this count."
        ),
    )
    expected_samples = int(expected_samples_value) if expected_samples_value else None

    f1, f2 = st.columns(2)
    source_upload = f1.file_uploader(
        f"Source / short-gradient table for {dataset}", type=["csv", "txt", "tsv", "xlsx", "xls"], key=f"src_{i}"
    )
    target_upload = f2.file_uploader(
        f"Target / long-gradient table for {dataset}", type=["csv", "txt", "tsv", "xlsx", "xls"], key=f"tgt_{i}"
    )
    if (source_upload is None) != (target_upload is None):
        partial_uploads.append(dataset)
    if source_upload is not None and target_upload is not None:
        source_name = f"{dataset}__source{Path(source_upload.name).suffix or '.csv'}"
        target_name = f"{dataset}__target{Path(target_upload.name).suffix or '.csv'}"
        uploaded_files[source_name] = source_upload
        uploaded_files[target_name] = target_upload
        specs.append(
            DatasetSpec(
                dataset=dataset,
                matrix=sample_type,
                mode=mode,
                source_file=source_name,
                target_file=target_name,
                technical_replicates=technical_replicates,
                expected_biological_samples=expected_samples,
            )
        )

st.header("5. Optional verification files")
col1, col2 = st.columns(2)
auth_files_upload = col1.file_uploader(
    "Authentic-standard files", type=["csv", "txt", "tsv", "xlsx", "xls"], accept_multiple_files=True
)
iso_files_upload = col2.file_uploader(
    "Isotope-labeled standard files", type=["csv", "txt", "tsv", "xlsx", "xls"], accept_multiple_files=True
)
st.info("The app infers ionization mode from standard filenames containing POS or NEG.")

st.header("6. Run")
st.write(f"Models: **{len(selected_models)}** | Fast mode: **{'ON' if fast_mode else 'OFF'}** | Complete datasets ready: **{len(specs)}**")
st.code(", ".join(selected_models), language="text")

if partial_uploads:
    st.warning("These datasets have only one of the two required feature tables: " + ", ".join(partial_uploads))

run_clicked = st.button("Run RTBridge analysis", type="primary")
if run_clicked:
    if not specs:
        st.error("Upload at least one complete source/target feature-table pair.")
        st.stop()
    names = [s.dataset for s in specs]
    if len(names) != len(set(names)):
        st.error("Dataset names must be unique. Please rename duplicate datasets before running.")
        st.stop()

    run_root = Path(tempfile.mkdtemp(prefix="rtbridge_app_"))
    workdir = run_root / "input"
    output_dir = run_root / "RTBridge_outputs"
    workdir.mkdir(parents=True, exist_ok=True)

    progress_text = st.empty()
    progress_bar = st.progress(0)
    total_steps = max(1, len(specs) * 2 + len(specs) * 4 * len(selected_models) + 3)
    completed_steps = 0

    def progress(message: str):
        nonlocal_state["completed"] += 1
        progress_text.write(message)
        progress_bar.progress(min(0.96, nonlocal_state["completed"] / total_steps))

    nonlocal_state = {"completed": completed_steps}

    try:
        for filename, uploaded in uploaded_files.items():
            (workdir / filename).write_bytes(uploaded.getbuffer())

        auth_map = {}
        for uploaded in auth_files_upload or []:
            (workdir / uploaded.name).write_bytes(uploaded.getbuffer())
            lower = uploaded.name.lower()
            if "neg" in lower:
                auth_map["NEG"] = uploaded.name
            elif "pos" in lower:
                auth_map["POS"] = uploaded.name

        iso_map = {}
        for uploaded in iso_files_upload or []:
            (workdir / uploaded.name).write_bytes(uploaded.getbuffer())
            lower = uploaded.name.lower()
            if "neg" in lower:
                iso_map["NEG"] = uploaded.name
            elif "pos" in lower:
                iso_map["POS"] = uploaded.name

        result = run_rtbridge(
            specs,
            workdir,
            output_dir,
            settings,
            authentic_files=auth_map,
            isotope_files=iso_map,
            progress=progress,
        )
        progress_text.write("Generating manuscript and Supporting Information figures/tables")
        make_all_figures_tables(
            output_dir,
            dpi=settings.output_dpi,
            output_formats=("png", "pdf") if fast_mode else ("png", "tif", "pdf"),
            include_exploratory=not fast_mode,
        )
        progress_bar.progress(0.98)

        zip_name = f"RTBridge_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = run_root / zip_name
        zip_output(output_dir, zip_path)

        table2_path = output_dir / "tables" / "Table_2_primary_PCHIP_direct_comparison.csv"
        candidate_summary_path = output_dir / "tables" / "All_candidate_pair_counts_by_model.csv"
        parser_report_path = output_dir / "results" / "standard_file_parse_report.csv"
        aggregation_report_path = output_dir / "results" / "biological_sample_aggregation_report.csv"

        st.session_state.rtbridge_last_run = {
            "zip_name": zip_name,
            "zip_bytes": zip_path.read_bytes(),
            "dataset_count": len(specs),
            "model_count": len(selected_models),
            "model_result_count": len(result["results"]),
            "primary_pair_count": len(result["pairs"]),
            "candidate_pair_count": len(result.get("candidate_pairs", [])),
            "metrics_preview": result["results"].head(100).copy(),
            "pairs_preview": result["pairs"].head(100).copy(),
            "feature_summary": result["feature_summary"].copy(),
            "table2": pd.read_csv(table2_path) if table2_path.exists() else pd.DataFrame(),
            "candidate_summary": pd.read_csv(candidate_summary_path).head(200) if candidate_summary_path.exists() else pd.DataFrame(),
            "parser_report": pd.read_csv(parser_report_path) if parser_report_path.exists() else pd.DataFrame(),
            "aggregation_report": pd.read_csv(aggregation_report_path) if aggregation_report_path.exists() else pd.DataFrame(),
            "auth_zero_warning": bool(auth_map) and result["results"].get("Authentic_hits", pd.Series(dtype=float)).sum() == 0,
            "iso_zero_warning": bool(iso_map) and result["results"].get("Isotope_hits", pd.Series(dtype=float)).sum() == 0,
            "fast_mode": fast_mode,
        }
        progress_bar.progress(1.0)
        progress_text.write("Completed")
        st.success("RTBridge analysis completed successfully.")
    except Exception as exc:
        st.error("RTBridge analysis failed.")
        st.exception(exc)
        st.warning("Check the m/z, RT, biological-sample, replicate naming/count, and QC columns. QC columns should contain 'QC' or 'pool'.")
    finally:
        shutil.rmtree(run_root, ignore_errors=True)

last_run = st.session_state.rtbridge_last_run
if last_run:
    st.divider()
    st.subheader("Latest completed run")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", last_run["dataset_count"])
    c2.metric("Models", last_run["model_count"])
    c3.metric("Model-result rows", last_run["model_result_count"])
    c4.metric("Primary pairs", last_run["primary_pair_count"])
    if not last_run["fast_mode"]:
        st.metric("Exported candidate-pair rows", last_run["candidate_pair_count"])

    st.download_button(
        "Download complete RTBridge output ZIP",
        data=last_run["zip_bytes"],
        file_name=last_run["zip_name"],
        mime="application/zip",
        type="primary",
    )

    st.subheader("Model metrics preview")
    st.dataframe(format_publication_table(last_run["metrics_preview"]), use_container_width=True)
    if not last_run["pairs_preview"].empty:
        st.subheader("Primary pairs preview")
        preferred = [
            "Dataset", "Workflow", "Model", "Anchor_configuration", "Source_row", "Target_row",
            "Source_mz", "Target_mz", "Source_RT", "Target_RT", "ACS", "Verified_any",
            "Authentic_standard_names", "Isotope_standard_names", "Verified_metabolite_names",
        ]
        cols = [c for c in preferred if c in last_run["pairs_preview"].columns]
        st.dataframe(format_publication_table(last_run["pairs_preview"][cols]), use_container_width=True)

    st.subheader("Feature and anchor summary")
    st.dataframe(format_publication_table(last_run["feature_summary"]), use_container_width=True)
    if not last_run["aggregation_report"].empty:
        st.subheader("Biological-sample and technical-injection grouping")
        st.caption("Review this table to confirm that each independent sample and its replicate injections were recognized correctly.")
        st.dataframe(last_run["aggregation_report"], use_container_width=True)
    if not last_run["table2"].empty:
        st.subheader("Paper Table 2 preview")
        st.dataframe(format_publication_table(last_run["table2"]), use_container_width=True)
    if not last_run["candidate_summary"].empty:
        st.subheader("Candidate-pair summary by model")
        st.dataframe(format_publication_table(last_run["candidate_summary"]), use_container_width=True)
    if last_run["auth_zero_warning"]:
        st.warning("Authentic-standard files were uploaded, but no authentic hits were found. Review the parser report and verification windows.")
    if last_run["iso_zero_warning"]:
        st.warning("Isotope-labeled files were uploaded, but no isotope hits were found. Review the parser report and verification windows.")
    if not last_run["parser_report"].empty:
        st.subheader("Standard-file parser report")
        st.dataframe(last_run["parser_report"], use_container_width=True)
