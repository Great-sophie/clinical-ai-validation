# Final Results

> These results use the repository's **synthetic example prediction cohorts** and demonstrate the validation workflow. They are not clinical performance claims.

## Internal vs external validation

| Metric | Internal | External |
|---|---:|---:|
| N | 300 | 220 |
| Prevalence | 0.3500 | 0.2955 |
| AUROC (95% CI) | 0.9496 (0.9187-0.9731) | 0.9097 (0.8596-0.9498) |
| AUPRC (95% CI) | 0.9269 (0.8841-0.9594) | 0.8368 (0.7490-0.9066) |
| Brier (95% CI) | 0.1266 (0.1101-0.1438) | 0.1564 (0.1358-0.1779) |
| Sensitivity (95% CI) | 0.9524 (0.9091-0.9905) | 0.9077 (0.8333-0.9702) |
| Specificity (95% CI) | 0.7795 (0.7150-0.8342) | 0.7677 (0.7000-0.8301) |
| PPV (95% CI) | 0.6993 (0.6204-0.7762) | 0.6211 (0.5232-0.7143) |
| NPV (95% CI) | 0.9682 (0.9375-0.9937) | 0.9520 (0.9113-0.9841) |
| F1 (95% CI) | 0.8065 (0.7454-0.8582) | 0.7375 (0.6543-0.8072) |
| Calibration intercept (95% CI) | -1.4881 (-2.0838--1.0967) | -1.4359 (-1.9672--1.0151) |
| Calibration slope (95% CI) | 2.2475 (1.8619-2.7984) | 1.8620 (1.4394-2.4492) |

## External recalibration

Recalibration parameters were fitted on the internal/calibration cohort and then frozen before application to the external cohort.

| Metric | Before | After |
|---|---:|---:|
| AUROC | 0.9097 | 0.9097 |
| AUPRC | 0.8368 | 0.8368 |
| Brier | 0.1564 | 0.1040 |
| Sensitivity @ 0.5 | 0.9077 | 0.7231 |
| Specificity @ 0.5 | 0.7677 | 0.8968 |
| PPV @ 0.5 | 0.6211 | 0.7460 |
| NPV @ 0.5 | 0.9520 | 0.8854 |
| F1 @ 0.5 | 0.7375 | 0.7344 |
| Calibration intercept | -1.4359 | -0.1832 |
| Calibration slope | 1.8620 | 0.8743 |

The monotonic recalibration preserves ranking, so AUROC/AUPRC remain unchanged, while Brier score improves from 0.1564 to 0.1040. Calibration moves substantially toward the ideal intercept 0 / slope 1.

## DCA threshold table with paired nested-bootstrap 95% CI

| Threshold | NB before | NB after | Delta NB | 95% CI for Delta NB | Delta NB / 100 |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.2222 | 0.2500 | 0.0278 | 0.0066-0.0460 | 2.78 |
| 0.20 | 0.1443 | 0.2295 | 0.0852 | 0.0557-0.1080 | 8.52 |
| 0.30 | 0.0955 | 0.1987 | 0.1032 | 0.0630-0.1416 | 10.32 |
| 0.40 | 0.0758 | 0.1788 | 0.1030 | 0.0500-0.1485 | 10.30 |
| 0.50 | 0.1045 | 0.1409 | 0.0364 | -0.0136-0.0909 | 3.64 |
| 0.60 | 0.0932 | 0.1136 | 0.0205 | -0.0250-0.0773 | 2.05 |

Bootstrap support for improved net benefit is strongest from thresholds 0.10-0.40 because the 95% CI for Delta NB remains above zero. At 0.50 and 0.60 the interval crosses zero, so improvement is less certain.
