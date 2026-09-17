# RTBridge

RTBridge is a runnable workflow for retention-time bridging across LC-HRMS gradients.

**Default behavior: Correlation, ratio error, ratio score, and ACS are computed from independent biological-sample-level summaries obtained by averaging technical replicate injections for each biological sample.**

In the present manuscript study, the independent biological samples are four donors. The same rule applies to non-human or non-donor studies: technical injections are averaged within each independent biological sample before scoring, and repeat injections are never treated as independent biological observations.

## Current-study preset

The browser interface opens with the present study as the starting preset:

- two sample types: plasma and whole blood
- positive and negative ionization modes, giving four datasets
- four independent biological samples per dataset
- three technical injections per biological sample
- one pooled-QC sample per matrix, planned for three injections per method; the supplied processed tables contain two or three usable QC columns, and RTBridge uses all QC columns actually present

QC columns are recognized separately by names containing `QC` or `pool`; they are used for QC mean/CV calculations and are not included in the biological-sample count. These values are editable starting points, not software limits.

## Flexible sample and replicate support

RTBridge supports:

- multiple sample types in one run, including plasma, whole blood, serum, urine, cerebrospinal fluid, saliva, stool/feces, tissue, cell pellets, culture media, bile, dried blood spots, and custom sample types
- any number of independent biological samples per sample type
- any consistent number of technical injections per biological sample
- different replicate counts for different datasets
- any number of pooled-QC injections, recognized separately from biological samples
- positive mode, negative mode, and custom modes

For example, a future study could contain three sample types, 10 independent biological samples per type, and five technical injections per sample. Each corresponding feature table would contain 50 biological injection columns plus any QC columns. RTBridge would average the five injections within each of the 10 biological samples and then calculate correlation, ratio error, ratio score, and ACS across the 10 independent sample-level means.

### Automatic recognition

Use consistent column names such as:

- `Sample01_rep1`, `Sample01_rep2`, ..., `Sample01_rep5`
- `Sample01_injection_1_POS`
- `Plasma_S01_R1_NEG`
- `P1-1-NEG`, `P1-2-NEG`, ..., `P1-5-NEG`

When names clearly identify the biological sample and injection number, choose **Auto-detect from column names**.

When names do not encode replicate groups, provide the expected number of technical injections per biological sample in the dataset panel. RTBridge then groups consecutive sample-major columns. For the current study, groups of three are used; in another study, groups of five or another count may be selected. The optional expected biological-sample count validates the layout. This input controls recognition only: averaging of technical injections is always applied.

The generated `biological_sample_aggregation_report.csv` records every recognized group, the raw injection columns, the grouping method, and the number of injections used.

## Run on Windows

For the first run, extract the ZIP and double-click `INSTALL_AND_RUN_RTBridge.bat`. The launcher creates a local `.venv`, installs the required packages, opens the browser, and starts the Streamlit application.

For later runs, double-click `RUN_RTBridge.bat`.

## Publication defaults

- Within-method uniqueness window: 20 ppm
- Cross-method anchor-matching window: 10 ppm
- Final candidate-matching window: 10 ppm
- Technical replicate injections are averaged within each biological sample before scoring
- Positive-correlation filter: r > 0
- Primary anchor configurations: mode-specific NEG/POS anchors for PL NEG, PL POS, WB NEG, and WB POS
- COMBINED POS/NEG anchors: evaluated and exported as a sensitivity analysis only

## Included files

- `app.py`: browser application
- `rtbridge/core.py`: analysis engine
- `rtbridge/plotting.py`: figures and tables
- `INSTALL_AND_RUN_RTBridge.bat`: first-run installer and launcher
- `RUN_RTBridge.bat`: normal launcher
- `INSTALL_DEPENDENCIES.bat`: dependency installation only
- `run_rtbridge_cli.py`: optional command-line interface
- `PUBLICATION_SETTINGS.json`: publication settings record
- `requirements.txt`: Python dependencies
- `examples/example_manifest.csv`: editable CLI manifest template

## ACS correlation penalty and manual weights

The browser interface provides two independent controls:

1. **Correlation penalty**
   - **Linear positive r:** uses the positive Pearson correlation directly.
   - **Squared positive r:** uses r² and therefore penalizes weak and moderate positive correlations more strongly.
2. **Manual ACS weights**
   - The correlation and ratio-score weights can be entered separately.
   - The software normalizes the two non-negative weights to sum to one.

The processing order is fixed and cannot be disabled:

1. Average technical replicate injections within each independent biological sample.
2. Calculate Pearson r across the biological-sample means.
3. Delete pairs with undefined, zero, or negative r.
4. Apply the selected linear-r or squared-r transformation to the remaining positive correlations.
5. Combine the transformed correlation score with the QC-normalized ratio score using the selected weights.

The selected penalty and weights are applied end-to-end. They affect anchor filtering and final candidate ranking and can therefore change anchors, RT models, candidate pairs, primary links, and verification results. The publication default remains **linear positive r with 0.5/0.5 weights**.


## Final publication rerun

### Exact spline/GAM implementation and screening terminology

The software label `Smoothing spline/GAM` denotes a SciPy cubic `UnivariateSpline`, with `s = max(1e-6, 0.01 × n × variance(target RT))`. The `Monotone GAM` first fits non-decreasing isotonic regression and then applies a cubic `UnivariateSpline`, with `s = max(1e-6, 0.005 × n × variance(isotonic target RT))`. These are GAM-like spline implementations; they are not `mgcv` or `pyGAM` models.

Two anchor sets are compared. **All selected anchors** include every anchor satisfying the mass-uniqueness, cross-method mass-agreement, pooled-QC, positive-correlation, and abundance-concordance criteria, without additional residual-based removal. **GAM-screened anchors** are the selected anchors remaining after iterative GAM-based residual screening. In each screening iteration, the software fits a cubic smoothing-spline/GAM RT trend, removes anchors whose absolute residual exceeds the selected multiplier times the mean absolute residual, and refits. The resulting GAM-screened anchors are then applied to all six final RT-transfer models. The output-table workflow values are `All selected anchors` and `GAM-screened anchors`.

The packaged `publication_outputs_20_10_10` folder contains the final compact outputs used to update the revised manuscript and Supporting Information. The primary analysis uses mode-specific anchor configurations for all four datasets.

- PCHIP using all selected mode-specific anchors: 1,518 training anchors, 5,595 primary pairs, mean median RT error 0.0232 min, mean median ACS 0.869, and 161 standard-supported verification hits.
- PCHIP using GAM-screened mode-specific anchors: 1,185 training anchors, 4,867 primary pairs, mean median RT error 0.0212 min, mean median ACS 0.874, and 151 verification hits.
- COMBINED anchors remain available for sensitivity analysis and are summarized separately in Table S3.

Publication figures and tables display every ACS and RT-error value with three
significant figures (for example, ACS 0.869 and RT error 0.0330 min). Figures 2
and 4 group the original model names under mode-specific and
COMBINED anchor headings, and every Figure S4 sensitivity bar includes
its numeric value.

## Reproduce the packaged current-study analysis

From a terminal opened in the extracted software folder, run:

```bash
python run_rtbridge_cli.py current_study_manifest.csv \
  --workdir . \
  --output RTBridge_outputs_20_10_10 \
  --u 20 --a 10 --c 10 \
  --auth-pos example_current_study_data/authentic_standards_POS.csv \
  --auth-neg example_current_study_data/authentic_standards_NEG.csv \
  --iso-pos example_current_study_data/isotope_labeled_POS.csv \
  --iso-neg example_current_study_data/isotope_labeled_NEG.csv
```
