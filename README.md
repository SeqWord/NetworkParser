# NetworkParser

NetworkParser is a research pipeline for microbial genomics. It takes per-sample VCFs (or a feature matrix) plus labelled metadata and trains classifiers for **one label** or an **ordered hierarchy** such as lineage → AMR → resistance profile.

In hierarchy mode, each child model is trained only on samples that follow its parent branch. Query then walks that same path, so each prediction is tied to the markers and route that produced it.

Typical lifecycle:

```text
labelled training data → train-hierarchy → model registry + .npb bundle
                                          → query new samples
                                          → evaluate / evaluate-hierarchy
                                          → optional cross-validate / annotate-panels
```

Training does the filtering, feature-panel selection, and model fitting. Query reuses that trained feature space. It does not rerun association tests or retrain models.

> NetworkParser is a research tool. Its predictions are not validated clinical diagnoses.

## 1. Install

```bash
git clone https://github.com/Nomlie/network_parser.git
cd network_parser
conda env create -f environment.yml
conda activate networkparser
```

Run every command from the repository root, using `run_network_parser.py`:

```bash
python run_network_parser.py --help
python run_network_parser.py train-hierarchy --help
python run_network_parser.py query --help
python run_network_parser.py evaluate --help
```

`--help` lists required inputs and config. Add `--more` for the full option list.

Before a run starts, the program checks arguments, an optional `--config` JSON file, input paths, and VCF counts. A VCF training folder needs at least 10 files by default (`min_sample_presence`). If something is wrong, it prints a numbered list of problems and exits.

## 2. Inputs

You need genomic data, matching metadata, a reference genome for VCF, FASTA, or FASTQ, and (for query) a trained model. Settings for a repeatable run live in [`data/config.json`](data/config.json). The trained model is written to [`model/`](model/).

### Genomic data

Use either:

- a folder with **one VCF or gVCF per sample** (`*.vcf`, `*.vcf.gz`, including `.g.vcf.gz`), sitting directly in that folder; or
- a CSV/TSV **feature matrix** with sample IDs in the first column.

Example matrix:

```csv
Sample,chr1:761155:C:T,chr1:2155168:C:T
sample_001,0,1
sample_002,1,NaN
```

What the values mean:

| Value | Meaning |
|---|---|
| `0` | callable reference / baseline |
| `1` | callable alternate / non-baseline |
| `NaN` | missing or unresolved; never treated as reference |

Feature IDs look like `Contig:Pos:Ref:Alt`.

### Metadata

CSV or TSV. Include a sample-ID column (`Sample` if you have one, otherwise the first column) and the labels you want to predict.

```csv
Sample,Lineage_clean,AMR_binary
sample_001,lineage 4,susceptible
sample_002,lineage 2,resistant
```

Sample IDs must match VCF file names without the `.vcf` / `.vcf.gz` suffix. Duplicate IDs and extra spaces will break alignment.

### Reference genome

Use `--ref_fasta` with a FASTA or GenBank file for VCF, FASTA, or FASTQ. Training and query must use the **same build, contig names, coordinates, and allele orientation**.

The demo uses H37Rv (`data/reference/H37Rv.fasta` / `H37Rv.gbk`). AFRO demo VCFs use contig `M.tuberculosis_H37Rv`.

### Demo data

`data/` is a small AFRO-TB subset for trying the pipeline (150 train / 30 test VCFs, no shared sample IDs). Resistance labels here come from genotype/catalogue calls, not from independent phenotypic DST. The paper used a larger cohort. See [`data/README.md`](data/README.md).

### Config file

Most settings are command-line flags. For a repeatable experiment, put them in JSON and pass `--config`. A typical file is [`data/config.json`](data/config.json):

```json
{
  "qual_threshold": 30.0,
  "min_dp_per_sample": 10,
  "min_gq_per_sample": 0,
  "assume_absent_variant_is_reference": true,
  "n_jobs": -1,
  "random_state": 42,
  "central_feature_filter_method": "rf_fdr",
  "rf_selector_fallback_strategy": "stop",
  "feature_panel_threshold_failure_strategy": "stop"
}
```

This example matches variant-only VCF cohorts (GQ often missing; absent sites treated as reference). Edit the file rather than copying a new JSON blob into every command.

The defaults stop the run if FDR keeps no features or no panel reaches the score threshold. Fallbacks such as `top_n`, `unfiltered`, and `best_available` have to be turned on on purpose.

**Known-marker seed** (WHO-style catalogue forced into AMR/profile panels) is off in `data/config.json` until you enable it. See [`docs/KNOWN_MARKER_SEED.md`](docs/KNOWN_MARKER_SEED.md). The model deposited in `model/` was trained with that priority on. `data/config.json` is the small-demo default (`central_feature_filter_method` `rf_fdr`). It does not reproduce the deposited model.

All fields: [`network_parser/config.py`](network_parser/config.py).

### Trained model

The repository ships a trained AFRO hierarchy model in [`model/`](model/):

| File | Use |
|---|---|
| `model/networkparser_model_bundle.npb` | `--bundle` for query |
| `model/hierarchical_model_registry.json` | `--registry` for query, annotate, or `bundle` |

Hierarchy: `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed`. This bundle is the manuscript model: chi-square/Fisher false-discovery-rate filtering, known-mutation priority on, 10,974 AFRO-TB training genomes (`Hierarchy_Lineage_AMR_Resistance_Profile_seeded_01`). Logistic regression was selected for ten fitted nodes and random forest for three. The training configuration is [`model/afro_tb_seeded_training_config.json`](model/afro_tb_seeded_training_config.json).

Query with `--bundle model/networkparser_model_bundle.npb`. The registry records the training layout; the matrices and catalogues it names are not in this repository. `train-hierarchy` with `--output_dir model` replaces these files.

`.npb` files contain Python pickle objects. Load them only from this repository or another trusted training run.

## 3. How the pipeline works

### Hierarchical training

List labels from broadest to most specific. Each node is a model trained only on samples that reached that parent:

```text
Lineage_clean
     ├─ lineage 1 → AMR_binary → Resistance_Profile
     ├─ lineage 2 → AMR_binary → Resistance_Profile
     └─ …
```

At each node, training:

1. Builds the sample × variant matrix with the shared VCF call rules
2. Applies cohort filters (presence, missingness, invariant/minor-count)
3. Runs **central feature filtering** (default RF-FDR, or chi-square / Fisher FDR)
4. Picks a compact **feature panel** using a separability check
5. Fits the ML protocol / model selector, and a decision tree when that tree is competitive
6. Writes the node outputs (model, selected features, ranked tables, summaries)

Query follows the same path. It records support scores, fallbacks, and weak-evidence flags. If too few trained markers are recovered, it can withhold a class call.

### Single-label training

`run` trains one metadata column with the same matrix and filters, without the hierarchy tree.

### Callability (same rules in train and query)

Variant-only VCFs often have no GQ and no reference sites. By default a site missing from that VCF is **unknown**. Set `assume_absent_variant_is_reference` if you want missing sites treated as reference.

| Config | Role |
|---|---|
| `qual_threshold`, `min_dp_per_sample`, `min_gq_per_sample` | Per-call QC |
| `assume_absent_variant_is_reference` | Opt-in: missing site → REF (legacy variant-only cohorts) |
| `expand_gvcf_ref_blocks` | Treat gVCF REF blocks as callable reference |
| `min_feature_recovery_fraction` | Query gate: fraction of the trained panel recovered |
| `min_callable_fraction` | Query gate: fraction of recovered sites that are callable |

If a query sample falls below those gates, the result is review/abstention instead of a class call.

## 4. Run the pipeline

The examples use the demo data and the shipped model in `model/`. Swap in your own paths for a real cohort. Retraining with default RF-FDR settings can take a long time; query does not need a retrain.

### Train a hierarchy

```bash
python run_network_parser.py train-hierarchy \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary \
  --ref_fasta data/reference/H37Rv.fasta \
  --config data/config.json \
  --output_dir model \
  --n_jobs -1
```

This writes `model/hierarchical_model_registry.json` and `model/networkparser_model_bundle.npb`.

If your metadata already uses these column names, you can pass a preset:

| `--hierarchy_preset` | Columns |
|---|---|
| `lineage_amr_binary` | `Lineage_clean` → `AMR_binary` |
| `lineage_amr_profile` | `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed` |
| `lineage_family_amr_profile` | `Lineage_family` → `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed` |

```bash
python run_network_parser.py train-hierarchy \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --hierarchy_preset lineage_amr_profile \
  --ref_fasta data/reference/H37Rv.gbk \
  --config data/config.json \
  --output_dir model
```

If you set both, `--hierarchy_labels` is used.

### Train one label

```bash
python run_network_parser.py run \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --label Lineage_clean \
  --ref_fasta data/reference/H37Rv.fasta \
  --config data/config.json \
  --output_dir model \
  --n_jobs -1
```

### Query new samples

```bash
python run_network_parser.py query \
  --genomic data/test \
  --bundle model/networkparser_model_bundle.npb \
  --query_input_type auto \
  --ref_fasta data/reference/H37Rv.fasta \
  --config data/config.json \
  --output_dir results/query \
  --n_jobs -1
```

| `--query_input_type` | Input |
|---|---|
| `auto` | Detect from the path (default) |
| `vcf` | One VCF or a directory of VCFs |
| `matrix` | CSV/TSV in the training feature space |
| `fasta` | FASTA sequence |
| `fastq` | Directory of paired-end FASTQ files (BWA + bcftools; optional panel calling) |

You can pass `--registry model/hierarchical_model_registry.json` instead of `--bundle`. The bundle is the portable file for query.

### Evaluate

One label:

```bash
python run_network_parser.py evaluate \
  --predictions results/query/query_predictions.csv \
  --meta data/test_metadata.csv \
  --label AMR_binary \
  --output_dir results/evaluation
```

Full hierarchy (per-level metrics, path accuracy, bootstrap CIs):

```bash
python run_network_parser.py evaluate-hierarchy \
  --predictions results/query/query_predictions.csv \
  --meta data/test_metadata.csv \
  --hierarchy_labels Lineage_clean AMR_binary \
  --output_dir results/hierarchy_evaluation
```

### Leakage-aware cross-validation

`cross-validate` scores **one** metadata column at a time. Feature filtering and panel selection happen inside each training fold.

```bash
python run_network_parser.py cross-validate \
  --genomic data/train \
  --meta data/train_metadata.csv \
  --label AMR_binary \
  --config data/config.json \
  --output_dir results/cv \
  --n_repeats 3 \
  --n_splits 5
```

For a hierarchy, run CV once per label column, on training samples only. Keep held-out query samples out of CV.

### Annotate selected panels

Adds gene names, predicted consequences, and optional catalogue labels to the panels already chosen in training. It does not pick new features.

```bash
python run_network_parser.py annotate-panels \
  --registry model/hierarchical_model_registry.json \
  --output_dir results/annotation \
  --catalogue path/to/resistance_catalogue.tsv
```

## 5. Commands

```bash
python run_network_parser.py <command> --help
```

Add `--more` for the full option list.

| Command | Role |
|---|---|
| `train-hierarchy` | Multi-level, parent-scoped models (alias: `train-two-level`) |
| `query` | Infer on matrix / VCF / FASTA / FASTQ |
| `evaluate` | Score one label against metadata |
| `evaluate-hierarchy` | Hierarchy evaluation pack |
| `run` | Single-label training |
| `bundle` | Package a registry into a portable `.npb` |
| `cross-validate` | Repeated nested CV for one label |
| `annotate-panels` | Gene / catalogue context on selected panels |

## 6. Outputs

| File | Produced by | Purpose |
|---|---|---|
| `model/hierarchical_model_registry.json` | `train-hierarchy` | Hierarchy, node paths, features, fallbacks |
| `model/networkparser_model_bundle.npb` | `train-hierarchy` / `bundle` | Portable query artefact |
| `query_predictions.csv` | `query` | Full prediction table |
| `query_predictions_compact.tsv` | `query` | Compact table |
| `query_predictions_readable.html` | `query` | Human-readable report |
| `query_route_audit.json` | `query` | Route, fallbacks, weak-evidence flags |
| `query_alignment_summary.json` | `query` | Feature recovery / callability |
| `evaluation_summary.json` | `evaluate` | Per-label scores |

Each trained node folder also has the selected-feature list, ranked marker tables, and the model file.

## 7. Troubleshooting

**Samples do not align.** Sample IDs must match exactly between VCF names (minus `.vcf` / `.vcf.gz`) and metadata. Check duplicates and extra spaces.

**Many values are missing.** Check QUAL/DP/GQ/MQ, FILTER, ploidy, gVCF vs variant-only calling, and `assume_absent_variant_is_reference`.

**Query feature recovery is low.** Training and query must use the same reference, contig names, coordinates, and allele orientation. See `query_alignment_summary.json`.

**A hierarchy branch was skipped.** That node had too few samples or classes after the low-support filter. See the registry and `query_route_audit.json`. Rare classes can be reported as `low_support_review_required`.

**Weak-evidence susceptible AMR calls.** A branch AMR model can call susceptible when too few resistance markers resolve. By default the class label is kept and a warning is attached. `--amr_weak_evidence_mode block` replaces the call with a review label.

**Too few VCF files.** Training needs at least `min_sample_presence` VCFs (default 10) directly in `--genomic`. Nested folders are ignored.

## 8. Tests and further reading

```bash
pytest -q
```

- [Demo data](data/README.md)
- [Architecture overview](docs/NETWORKPARSER_FULL_PICTURE.md)
- [Known-marker seed](docs/KNOWN_MARKER_SEED.md)
