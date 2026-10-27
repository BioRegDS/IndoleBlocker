# Receptor-Aware Machine Learning for Virtual Screening of Broad-Spectrum Antagonists against Indole-Responsive Olfactory Receptors

<!-- Add the graphical abstract here when ready:
<p align="center">
  <img src="docs/graphical_abstract.png" width="500" alt="Graphical abstract">
</p>
-->

## Overview

This repository provides research inputs, Python scripts, and analysis outputs for receptor-aware prediction of antagonism at indole-responsive mouse olfactory receptors and virtual screening of fragrance-related compounds.

The study combines patent-derived antagonist IC50 data, receptor amino acid sequences, and molecular fingerprints. Regression models were compared using leave-one-receptor-out (LORO) cross-validation. The selected approach uses receptor amino acid composition (AAC), molecular fingerprints, and gradient boosting regression (GBR). Additional analyses assess receptor representation, permutation controls, feature interpretation, and receptor isoleucine composition.

The trained model was applied to an IFRA/PubChem-derived compound library to prioritize candidates for broad-spectrum antagonism. Known fragrance-related antagonists were evaluated as positive controls, and literature/patent evidence for the top-ranked candidates was annotated manually.

**Predictions and candidate rankings are computational findings, not experimental confirmation of antagonism.**

## Paper

- **Title:** Receptor-Aware Machine Learning for Virtual Screening of Broad-Spectrum Antagonists against Indole-Responsive Olfactory Receptors
- **Journal:** To be updated upon publication
- **DOI:** To be updated upon publication

## Requirements

The workflow uses Python 3.11, NumPy, pandas, SciPy, scikit-learn, RDKit, PyTorch, Transformers, Biopython, Matplotlib, requests, openpyxl, joblib, and tqdm. The principal versions are specified in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate odorblock
```

The environment specification successfully passed dependency resolution on Linux (`linux-64`). Installation and full execution in a newly created environment have not yet been independently tested. Receptor embedding experiments use the pretrained ESM-2 model `facebook/esm2_t6_8M_UR50D`; its weights must be available or downloaded where applicable.

## Usage

Run commands from the repository root with the `odorblock` conda environment active. The workflow is organized by stage; **it is not a single automated end-to-end command**. Inspect individual script options with `python path/to/script.py --help` and specify input and output paths as appropriate.

### 1. Dataset preparation

`src/01_dataset/` validates the curated patent-derived activity dataset and extracts the indole IC50 subset. The source dataset contains 113 curated observations; the indole subset comprises 37 antagonist–receptor observations involving nine antagonists and ten mouse olfactory receptors.

```bash
python src/01_dataset/01_validate_prepare_dataset.py \
  --input data/Supplementary_Table_S1_curated_patent_OR_antagonist_dataset.csv \
  --outdir results/dataset
```

### 2. Model validation

`src/02_model_validation/` contains LORO model comparison, sequence-representation controls, and AAC permutation testing:

- `01_train_indole_loro_pic50_modelselect.py`
- `02_train_indole_loro_sequence_control.py`
- `03_permutation_control_aac_loro_gbr.py`

The selected AAC + GBR model achieved an LORO RMSE of approximately 0.351 pIC50 units in the reported model comparison. Because model selection and performance estimation used the same LORO framework, this estimate is **not** performance on an independent external test set.

For ESM-2 comparisons, supply the receptor-sequence input and, when appropriate, `--embedding_cache`, `--esm_model`, and `--device`. The exact commands and outputs used in the reproduction checks should be documented in `QC_REPORT.md`.

### 3. Interpretation

`src/03_interpretability/` provides AAC importance and ablation analyses, and receptor isoleucine-response analyses:

- `01_analyze_indole_aac_interpretability.py`
- `02_analyze_isoleucine_response_phenotype.py`

### 4. Virtual screening

`src/04_virtual_screening/` contains:

- `01_download_ifra_pubchem_library.py` — retrieve IFRA/PubChem source records
- `02_prepare_ifra_screening_library.py` — standardize compounds and remove overlaps
- `03_virtual_screen_ifra_indole_aac_gbr.py` — generate receptor-level predictions and rankings
- `04_analyze_positive_control_enrichment.py` — assess known positive-control ranks

The workflow retrieved 2,514 records, standardized these to 2,500 unique compounds, and excluded 41 compounds overlapping with the full curated patent dataset, leaving 2,459 ranked candidates. Predictions were generated for 2,500 compounds across ten receptors (25,000 compound–receptor predictions) before ranking exclusions.

To reproduce the original positive-control enrichment settings, **explicitly** use 100,000 permutations, seed 42, and the original thresholds:

```bash
python src/04_virtual_screening/04_analyze_positive_control_enrichment.py \
  --result_dir results/virtual_screening/predictions \
  --n_permutations 100000 \
  --seed 42 \
  --top_percentages 10,20,30,40,50 \
  --outdir results/virtual_screening/positive_control_enrichment
```

The enrichment p-values are threshold-specific and have not been adjusted for multiple threshold comparisons. Check the actual screening result directory before executing this example.

### 5. External evidence annotation

`src/05_external_validation/01_update_top50_external_evidence_annotation.py` merges the current Top 50 candidate ranking with **previously manually curated** literature and patent evidence. The script does not search the literature, assess patents, or assign evidence grades. Evidence annotations are not independent experimental validation of predicted activity.

### 6. Figures

`src/figures/` contains scripts for computational panels and figure assembly. For example:

```bash
python src/figures/plot_figure5.py \
  --input results/external_validation/ifra_top50_external_evidence_annotation_latest.csv \
  --outdir figures/figure5
```

Figure 1A and Figure 4A were manually prepared in PowerPoint and are not regenerated by the Python scripts. Other panels were checked in the reproduction QC; minor rendering differences can arise from the plotting environment.

## Reproducibility

The packaged analysis scripts were run against the available research inputs, and regenerated tables were compared with the original analysis outputs in the working research environment. Checks covered dataset preparation, model comparisons, sequence controls, AAC permutations, interpretation, library preparation, virtual screening, positive-control enrichment, and external-annotation integration. The checked numerical tables matched the reference outputs. The Top 50 annotation table reproduced all 50 compounds and all 42 columns.

These checks establish agreement in the tested environment, **not** independent experimental validation, verification of every literature/patent assertion, or guaranteed cross-platform reproducibility. See `QC_REPORT.md` for the detailed QC record.

## DataAvailability

The antagonist activity dataset was curated from two publicly available patent publications (WO2019101821A2 and WO2019008089A1). Experimental activity data, compound structures, and receptor information were extracted and organized into a machine-readable dataset with assistance from ChatGPT. Source patent identifiers and locations are retained to ensure traceability.

The fragrance compound library was obtained from PubChem using compound records associated with the International Fragrance Association (IFRA) and subsequently standardized using RDKit.

The repository provides the analysis scripts, processed datasets, and computational results required to reproduce the study, subject to applicable data redistribution terms. Original source materials remain subject to the terms of their respective providers.

## Author

**BioRegDS**  
Data Science Unit, Department of Biological Regulation
