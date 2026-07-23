# Full walk-forward demo backtest

Selected by validation spread using 2023–2024 only: **quant_sec**.

| Variant | Validation IC | Validation spread | 2025 IC | 2025 spread | 2026 IC | 2026 spread |
|---|---:|---:|---:|---:|---:|---:|
| quant_base | +0.0212 | +0.38% | +0.0277 | +0.43% | +0.0183 | +0.68% |
| quant_sec | +0.0290 | +0.50% | +0.0252 | +0.44% | +0.0187 | +0.68% |
| quant_sec_macro | +0.0338 | +0.46% | +0.0221 | +0.37% | +0.0196 | +0.25% |

## Lower-turnover 20-session demo

Selected by the same validation rule: **quant_sec_20d**.

| Variant | Validation IC | Validation spread | 2025 IC | 2025 spread | 2026 IC | 2026 spread |
|---|---:|---:|---:|---:|---:|---:|
| quant_base_20d | +0.0521 | +1.87% | +0.0532 | +2.49% | +0.0244 | +3.20% |
| quant_sec_20d | +0.0577 | +2.29% | +0.0340 | +1.66% | +0.0333 | +3.31% |
| quant_sec_global_context_20d | +0.0576 | +2.29% | +0.0340 | +1.66% | +0.0333 | +3.28% |
| quant_sec_sector_specialist_20d | +0.0577 | +2.29% | +0.0340 | +1.66% | +0.0333 | +3.31% |

Selected long-horizon cost audit:

- 2025: gross +1.66%, 10 bps/side net +1.26%, HAC p=0.275.
- 2026: gross +3.31%, 10 bps/side net +2.91%, HAC p=0.244.

## Selected-model audit

- 2025: gross spread +0.44%, 10 bps/side net +0.04%, spread HAC p=0.181; paired IC lift versus quant base -0.0025 (HAC p=0.674).
- 2026: gross spread +0.68%, 10 bps/side net +0.28%, spread HAC p=0.312; paired IC lift versus quant base +0.0004 (HAC p=0.957).

The model is a tabular walk-forward Ridge ranker. Each row is one ticker/session; inputs are delayed to the next tradable session.

Important limitations: current-index survivorship bias, incomplete ticker/CIK lineage, and overlapping holding-period returns. Treat this as a prototype demo rather than evidence of deployable alpha.

Machine-readable details: `full_backtest_demo_summary.json`.
