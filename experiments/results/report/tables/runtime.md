**Runtime in seconds, retraining vs. Fair Unlearning (random deletion), ours vs. paper Table 2.**

| Method | COMPAS 5% | COMPAS 20% | Adult 5% | Adult 20% | HSLS 5% | HSLS 20% |
|---|---|---|---|---|---|---|
| Retraining (Fair) (ours) | 0.0264 ± 0.0086 | 0.0237 ± 0.0106 | 0.1052 ± 0.0119 | 0.1048 ± 0.0156 | 0.0429 ± 0.0114 | 0.0366 ± 0.0092 |
| Retraining (Fair) (paper) | 3.500 ± 0.340 | 3.180 ± 0.142 | 36.796 ± 1.090 | 31.175 ± 1.040 | 14.963 ± 0.396 | 12.362 ± 0.795 |
| Fair Unlearning (ours) | 0.0016 ± 0.0001 | 0.0014 ± 0.0002 | 0.0091 ± 0.0012 | 0.0082 ± 0.0001 | 0.0146 ± 0.0010 | 0.0124 ± 0.0021 |
| Fair Unlearning (paper) | 0.170 ± 0.010 | 0.164 ± 0.003 | 8.800 ± 0.511 | 8.483 ± 0.938 | 6.014 ± 0.249 | 5.704 ± 0.015 |
| Speedup (ours) | 16.9× | 16.4× | 11.6× | 12.8× | 2.9× | 2.9× |

Ours: CPU, float64; the Newton step computes the Hessian on the remaining data at unlearning time (no precomputation). Paper: 32GB V100 GPU (Appendix A.7, Table 2, whose first row is labelled only "Retraining"). Absolute times are not directly comparable across hardware; speedups also depend on implementation and Hessian precomputation.
