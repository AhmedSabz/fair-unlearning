**Runtime in seconds, retraining vs. Fair Unlearning (random deletion), ours vs. paper Table 2.**

| Method | COMPAS 5% | COMPAS 20% | Adult 5% | Adult 20% | HSLS 5% | HSLS 20% |
|---|---|---|---|---|---|---|
| Retraining (Fair) (ours) | 0.0212 ± 0.0071 | 0.0202 ± 0.0054 | 0.0850 ± 0.0043 | 0.0911 ± 0.0192 | 0.0354 ± 0.0031 | 0.0305 ± 0.0076 |
| Retraining (Fair) (paper) | — | — | 36.796 ± 1.090 | — | 14.963 ± 0.396 | 12.362 ± 0.795 |
| Fair Unlearning (ours) | 0.0012 ± 0.0000 | 0.0013 ± 0.0001 | 0.0083 ± 0.0003 | 0.0075 ± 0.0006 | 0.0129 ± 0.0004 | 0.0111 ± 0.0008 |
| Fair Unlearning (paper) | — | — | 8.800 ± 0.511 | — | 6.014 ± 0.249 | 5.704 ± 0.015 |
| Speedup (ours) | 17.8× | 16.1× | 10.3× | 12.1× | 2.7× | 2.7× |

Ours: CPU, float64; the Newton step computes the Hessian on the remaining data at unlearning time (no precomputation). Paper: 32GB V100 GPU. Absolute times are not comparable across hardware; compare the speedup. — = not reported or not legible in the published table.
