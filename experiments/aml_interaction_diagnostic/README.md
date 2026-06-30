# AML Interaction Diagnostic

This experiment studies whether AML explanations become less faithful on samples with strong local second-order word interactions.

The main diagnostic uses AML-compatible hard-deletion faithfulness semantics:

- attribution values are interpreted as retention strength;
- special tokens are not candidate word nodes;
- the interaction teacher score is target-class probability;
- dependency edges use spaCy when enabled.

## Smoke Test

```bash
python experiments/aml_interaction_diagnostic/scripts/run_diagnostic.py --max-samples 2 --disable-dependency
```

The smoke path uses a deterministic mock adapter so it can run on CPU without model checkpoints.

For detailed usage, including model paths, datasets, output directories, and GPU selection, see `USAGE.md`.

## Candidate Coverage

```bash
python experiments/aml_interaction_diagnostic/scripts/run_candidate_coverage.py --max-words 20 --num-samples 2
```

## Full Integration Note

`src/aml_adapter.py` contains the adapter boundary for the official AML implementation. A trained AML checkpoint should be exposed through the same `AttributionOutput` contract before running full datasets.
