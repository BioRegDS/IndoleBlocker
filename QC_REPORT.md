# QC review — first pass

## Completed

- Extracted 01–04 archives; removed editor backup files ending in `~`.
- Added external evidence synchronization script as stage 05.
- Confirmed 11 Python scripts pass Python syntax compilation.
- Preserved original script logic (no refactoring yet).

## Must resolve before release

1. **Source-data traceability:** obtain and compare Supplementary Table S1 with the actual manuscript modeling input. The dataset-preparation script writes `results/dataset/indole_ic50_dataset.csv`, while model scripts require receptor sequences. Verify their provenance and exact handoff.
2. **Library counts:** independently reproduce 2,514 -> 2,498 -> 2,457 and reconcile with previously observed 2,459 novel rows; do not modify manuscript counts until matched to final output files.
3. **Model selection and leakage:** audit every LORO fold and any model/hyperparameter selection using held-out receptor results. Confirm manuscript claims reflect the actual selection protocol.
4. **Figure 5:** the synchronization script requires a pre-existing manual annotation CSV; it cannot recreate grade assignments or citations on its own.
5. **Dependencies:** freeze actual package versions from the original analysis environment before creating a reliable `environment.yml`.
6. **Outputs:** run all scripts with original inputs and compare numeric metrics, ranks, compound counts and figure data to frozen manuscript outputs.
7. **Publication safety:** verify patent/IFRA/PubChem data redistribution conditions and inspect files for secrets or sensitive paths.

## Preliminary implementation observations

- Scripts use different output directories and CLI input conventions; a documented driver script or command manifest is needed.
- The dataset script checks unique record IDs, positive activities, pActivity recalculation and SMILES standardization.
- Syntax compilation is not a functional test; numerical correctness remains unverified.
