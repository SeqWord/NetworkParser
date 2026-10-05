# Trained NetworkParser model

This folder ships the AFRO-TB hierarchy reported in the manuscript.

| File | Use |
|---|---|
| `networkparser_model_bundle.npb` | `query --bundle` |
| `hierarchical_model_registry.json` | Node record. Query should use the bundle. |
| `afro_tb_seeded_training_config.json` | Settings used to train this bundle |

Hierarchy: `Lineage_clean` → `AMR_binary` → `Resistance_Profile_Collapsed`  
Experiment: `Hierarchy_Lineage_AMR_Resistance_Profile_seeded_01`  
Training genomes: 10,974.  
Screen: chi-square, or Fisher’s exact test when a 2×2 expected count is below five, then Benjamini–Hochberg false-discovery-rate control.  
Known-mutation priority: on at the phenotype-related stages.  
Fitted nodes: logistic regression for ten and random forest for three.

`data/config.json` is the small-demo default (random-forest false-discovery-rate filtering, known-mutation priority off). It does not reproduce this bundle.

Query new samples with:

```text
python run_network_parser.py query --bundle model/networkparser_model_bundle.npb
```

Paths in the registry are relative to the project root on the training machine. The training matrices and the resistance catalogue are not included here. `train-hierarchy` with `--output_dir model` replaces these files.

`.npb` files contain Python pickle objects. Load them only from this repository or another trusted training run.
