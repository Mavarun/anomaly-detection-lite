### Event F1 (mean ± std), cost per 1k, delay and point FAR; harm_dw3, static threshold, 10 seeds

| scenario | detector | event F1 | cost/1k | delay | point FAR |
| --- | --- | ---: | ---: | ---: | ---: |
| weekend | robust_z | 0.851 ± 0.077 | 11.1 | 3.3 | 0.0060 |
| weekend | cusum_k15 | 0.709 ± 0.140 | 22.0 | 13.9 | 0.0061 |
| weekend | cusum_pw | 0.759 ± 0.051 | 18.5 | 5.4 | 0.0035 |
| weekend | cusum_pwc | 0.802 ± 0.075 | 16.2 | 7.2 | 0.0050 |
| weekend_t5 | robust_z | 0.821 ± 0.059 | 12.7 | 5.4 | 0.0047 |
| weekend_t5 | cusum_k15 | 0.607 ± 0.288 | 24.2 | 14.7 | 0.0105 |
| weekend_t5 | cusum_pw | 0.760 ± 0.049 | 18.4 | 6.0 | 0.0060 |
| weekend_t5 | cusum_pwc | 0.692 ± 0.105 | 22.4 | 8.1 | 0.0049 |
| weekend_t3 | robust_z | 0.731 ± 0.179 | 18.2 | 8.3 | 0.0041 |
| weekend_t3 | cusum_k15 | 0.368 ± 0.292 | 34.7 | 26.9 | 0.0095 |
| weekend_t3 | cusum_pw | 0.694 ± 0.229 | 21.3 | 13.0 | 0.0127 |
| weekend_t3 | cusum_pwc | 0.578 ± 0.156 | 28.5 | 20.2 | 0.0086 |

### Paired per-seed effects, cusum_pwc - cusum_pw (10 seeds; bootstrap 95% CI; exact sign test)

| scenario | metric | mean diff | 95% CI | seeds with diff > 0 | sign p |
| --- | --- | ---: | :---: | ---: | ---: |
| weekend | event_f1 | +0.043 | [+0.017, +0.070] | 6/10 | 0.289 |
| weekend_t3 | event_f1 | -0.116 | [-0.219, +0.006] | 2/10 | 0.109 |
| weekend_t5 | event_f1 | -0.068 | [-0.111, -0.023] | 2/10 | 0.109 |
| weekend | cost_per_1k | -2.245 | [-4.210, -0.548] | 3/10 | 0.344 |
| weekend_t3 | cost_per_1k | +7.140 | [+2.562, +11.621] | 8/10 | 0.109 |
| weekend_t5 | cost_per_1k | +4.047 | [+1.854, +6.305] | 8/10 | 0.109 |

### Pre-registered decisions

{"H1": false, "H1_parts": {"t3_gain": false, "gauss_ok": true, "cost_ok": false}, "H2": false, "robust_z_f1": {"weekend": 0.8510422195988155, "weekend_t3": 0.7305965699142091, "weekend_t5": 0.8213229145654992}, "recommended_clip": null}
