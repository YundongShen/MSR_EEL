# Analysis notes

Commands for these analyses are in the
[reproduction guide](../reproduce/REPRODUCE.md#rq3-diagnostics-and-django-figures).

## Django figures and region statistics

The Django scripts select all 849 Django issues.

Region percentages are computed among retained T1/T2 hunks.
The per-hunk scores and run log are in
[`region_analysis.json`](../../Data/Evaluation/training_variants/region_analysis.json)
and [`region_analysis.log`](../../Data/Evaluation/training_variants/region_analysis.log).

## Hunk similarity

This analysis compares changed lines using UniXCoder cosine similarity.
Retained hunks are paired with reference hunks in the same file less than
20 lines apart. Pairing first uses unique references; the remaining 622 hunks
use the nearest same-file reference. T3 hunks use the nearest same-file
reference, with a closest-path fallback for one hunk. The random control uses
another issue in the same repository, with seed 0.

Commands are in the [reproduction guide](../reproduce/REPRODUCE.md#hunk-similarity).
