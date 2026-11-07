# Results report (config main)

Reference method: **fixed**. Paired Wilcoxon signed-rank test on total cost (same graphs), Holm-corrected over methods; r_rb = matched-pairs rank-biserial correlation (negative = cheaper than the reference); CI = 95% bootstrap interval of the median paired difference. Failed runs count as the budget.

## Cold start on test graphs (eval_final)

### p = 1, ALL (411 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 411 | 36.0 (20.0-60.0) | 326.1 | +24.0 [+24.0, +28.0] | +0.98 | 5% | 3.8e-63 |
| fixed | 411 | 12.0 (4.0-16.0) | 10.7 | reference | | | |
| sureshbabu | 411 | 12.0 (4.0-12.0) | 8.3 | +0.0 [+0.0, +0.0] | -0.54 | 41% | 6.1e-15 |
| ridge | 411 | 4.0 (4.0-4.0) | 4.2 | -8.0 [-8.0, -8.0] | -0.99 | 66% | 2.6e-48 |
| forest | 411 | 4.0 (4.0-4.0) | 8.1 | +0.0 [-8.0, +0.0] | -0.31 | 48% | 3e-06 |
| cache | 411 | 7.0 (7.0-7.0) | 7.0 | -5.0 [-5.0, -5.0] | -0.80 | 68% | 1.6e-45 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| random | 0.8220 | 0.432 | 90% |
| fixed | 0.9851 | 0.947 | 100% |
| sureshbabu | 0.9873 | 0.935 | 100% |
| ridge | 0.9991 | 0.986 | 100% |
| forest | 0.9998 | 0.965 | 100% |
| cache | 0.9998 | 0.994 | 100% |

Figure: `results/fig_cost_final_main_p1.png`

### p = 1, DVRP (187 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 187 | 36.0 (24.0-60.0) | 216.1 | +28.0 [+24.0, +36.0] | +1.00 | 1% | 3.7e-31 |
| fixed | 187 | 12.0 (4.0-12.0) | 8.2 | reference | | | |
| sureshbabu | 187 | 12.0 (12.0-12.0) | 12.0 | +0.0 [+0.0, +8.0] | +0.95 | 3% | 8e-20 |
| ridge | 187 | 4.0 (4.0-4.0) | 4.5 | +0.0 [-8.0, +0.0] | -0.96 | 46% | 8.6e-18 |
| forest | 187 | 12.0 (4.0-24.0) | 12.9 | +0.0 [+0.0, +4.0] | +0.90 | 6% | 3.9e-15 |
| cache | 187 | 7.0 (7.0-7.0) | 7.0 | -5.0 [-5.0, +3.0] | -0.51 | 51% | 3.7e-10 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| random | 0.8681 | 0.567 | 94% |
| fixed | 0.9900 | 0.975 | 100% |
| sureshbabu | 0.9539 | 0.935 | 100% |
| ridge | 0.9987 | 0.986 | 100% |
| forest | 0.9891 | 0.965 | 100% |
| cache | 0.9997 | 0.994 | 100% |

### p = 1, ER (112 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 112 | 46.0 (24.0-95.0) | 728.0 | +40.0 [+28.0, +52.0] | +1.00 | 1% | 8.2e-19 |
| fixed | 112 | 12.0 (4.0-12.0) | 9.6 | reference | | | |
| sureshbabu | 112 | 4.0 (4.0-12.0) | 6.5 | +0.0 [-8.0, +0.0] | -0.71 | 45% | 2.9e-07 |
| ridge | 112 | 4.0 (4.0-4.0) | 4.0 | -8.0 [-8.0, -8.0] | -1.00 | 66% | 3e-16 |
| forest | 112 | 4.0 (4.0-4.0) | 4.1 | -8.0 [-8.0, -8.0] | -1.00 | 66% | 2.7e-16 |
| cache | 112 | 7.0 (7.0-7.0) | 7.0 | -5.0 [-5.0, -5.0] | -0.77 | 66% | 7e-13 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| random | 0.8219 | 0.485 | 77% |
| fixed | 0.9860 | 0.962 | 100% |
| sureshbabu | 0.9915 | 0.975 | 100% |
| ridge | 0.9988 | 0.991 | 100% |
| forest | 0.9996 | 0.989 | 100% |
| cache | 0.9998 | 0.997 | 100% |

### p = 1, REG (112 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 112 | 24.0 (16.0-37.0) | 107.9 | +8.0 [+4.0, +12.0] | +0.85 | 15% | 9.6e-14 |
| fixed | 112 | 16.0 (16.0-16.0) | 15.8 | reference | | | |
| sureshbabu | 112 | 4.0 (4.0-4.0) | 4.1 | -12.0 [-12.0, -12.0] | -1.00 | 99% | 1.7e-24 |
| ridge | 112 | 4.0 (4.0-4.0) | 4.0 | -12.0 [-12.0, -12.0] | -1.00 | 100% | 1.7e-24 |
| forest | 112 | 4.0 (4.0-4.0) | 4.0 | -12.0 [-12.0, -12.0] | -1.00 | 100% | 1.7e-24 |
| cache | 112 | 7.0 (7.0-7.0) | 7.0 | -9.0 [-9.0, -9.0] | -1.00 | 100% | 1.7e-24 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| random | 0.7329 | 0.432 | 97% |
| fixed | 0.9491 | 0.947 | 100% |
| sureshbabu | 0.9994 | 0.989 | 100% |
| ridge | 0.9998 | 0.998 | 100% |
| forest | 1.0000 | 0.999 | 100% |
| cache | 1.0000 | 0.998 | 100% |

### p = 2, ALL (411 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 411 | 352.0 (140.0-3000.0) | 1540.9 | +320.0 [+216.0, +2912.0] | +0.99 | 3% | 2.1e-66 |
| fixed | 411 | 48.0 (24.0-64.0) | 48.4 | reference | | | |
| ridge | 411 | 8.0 (8.0-32.0) | 30.3 | -24.0 [-32.0, -16.0] | -0.77 | 70% | 6.5e-37 |
| forest | 411 | 8.0 (8.0-32.0) | 18.7 | -24.0 [-32.0, -24.0] | -0.91 | 75% | 8.7e-47 |
| cache | 411 | 11.0 (11.0-11.0) | 17.0 | -29.0 [-37.0, -29.0] | -0.93 | 83% | 2e-59 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| random | 0.7427 | 0.391 | 51% |
| fixed | 0.9534 | 0.800 | 100% |
| ridge | 0.9918 | 0.693 | 100% |
| forest | 0.9983 | 0.837 | 100% |
| cache | 0.9990 | 0.878 | 100% |

Figure: `results/fig_cost_final_main_p2.png`

### p = 2, DVRP (187 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 187 | 248.0 (136.0-3000.0) | 1319.9 | +208.0 [+176.0, +304.0] | +1.00 | 1% | 2.2e-31 |
| fixed | 187 | 32.0 (24.0-52.0) | 40.0 | reference | | | |
| ridge | 187 | 24.0 (8.0-48.0) | 46.0 | -8.0 [-16.0, +0.0] | -0.40 | 51% | 2.1e-05 |
| forest | 187 | 32.0 (8.0-48.0) | 30.2 | -8.0 [-16.0, +0.0] | -0.64 | 56% | 5.9e-10 |
| cache | 187 | 11.0 (11.0-35.0) | 23.9 | -13.0 [-13.0, -13.0] | -0.81 | 73% | 2e-21 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| random | 0.7589 | 0.490 | 59% |
| fixed | 0.9737 | 0.800 | 100% |
| ridge | 0.9847 | 0.693 | 99% |
| forest | 0.9831 | 0.837 | 100% |
| cache | 0.9937 | 0.878 | 100% |

### p = 2, ER (112 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 112 | 3000.0 (158.0-3000.0) | 1688.4 | +2928.0 [+264.0, +2944.0] | +1.00 | 0% | 2.3e-19 |
| fixed | 112 | 40.0 (24.0-50.0) | 38.9 | reference | | | |
| ridge | 112 | 8.0 (8.0-32.0) | 21.2 | -16.0 [-24.0, -16.0] | -0.82 | 73% | 1.6e-12 |
| forest | 112 | 8.0 (8.0-8.0) | 8.6 | -32.0 [-40.0, -28.0] | -1.00 | 81% | 1.2e-16 |
| cache | 112 | 11.0 (11.0-11.0) | 11.3 | -29.0 [-33.0, -21.0] | -0.94 | 83% | 1.5e-17 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| random | 0.7603 | 0.391 | 46% |
| fixed | 0.9624 | 0.900 | 100% |
| ridge | 0.9916 | 0.904 | 100% |
| forest | 0.9988 | 0.976 | 100% |
| cache | 0.9991 | 0.985 | 100% |

### p = 2, REG (112 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| random | 112 | 3000.0 (128.0-3000.0) | 1762.6 | +2912.0 [+151.9, +2920.0] | +0.96 | 11% | 5.2e-18 |
| fixed | 112 | 72.0 (64.0-80.0) | 71.9 | reference | | | |
| ridge | 112 | 8.0 (8.0-24.0) | 13.1 | -56.0 [-64.0, -56.0] | -1.00 | 98% | 1e-19 |
| forest | 112 | 8.0 (8.0-8.0) | 9.4 | -64.0 [-64.0, -56.0] | -1.00 | 99% | 1e-19 |
| cache | 112 | 11.0 (11.0-11.0) | 11.0 | -61.0 [-61.0, -53.0] | -1.00 | 100% | 8.9e-20 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| random | 0.6565 | 0.405 | 43% |
| fixed | 0.9389 | 0.864 | 100% |
| ridge | 0.9962 | 0.947 | 100% |
| forest | 0.9998 | 0.962 | 100% |
| cache | 1.0000 | 0.993 | 100% |

## Along the order stream (eval_stream)

### p = 1, ALL (160 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| fixed | 160 | 12.0 (12.0-24.0) | 16.7 | reference | | | |
| sureshbabu | 160 | 24.0 (24.0-28.0) | 25.9 | +12.0 [+12.0, +12.0] | +0.88 | 14% | 2.7e-19 |
| forest | 160 | 24.0 (4.0-32.0) | 18.6 | +0.0 [+0.0, +0.0] | +0.22 | 41% | 0.027 |
| temporal | 160 | 4.0 (4.0-4.0) | 4.4 | -8.0 [-10.0, -8.0] | -0.98 | 98% | 4.6e-27 |
| temporal+ML | 160 | 28.0 (4.0-32.0) | 20.8 | +8.0 [+0.0, +12.0] | +0.38 | 41% | 7.6e-05 |
| newton | 160 | 24.0 (24.0-24.0) | 24.0 | +12.0 [+10.0, +12.0] | +0.94 | 11% | 6.7e-22 |
| newton_reuseH | 160 | 8.0 (8.0-8.0) | 11.8 | -4.0 [-4.0, -4.0] | -0.57 | 81% | 1.7e-09 |
| select | 160 | 10.0 (10.0-10.0) | 13.7 | -2.0 [-2.0, -2.0] | -0.50 | 81% | 6.8e-08 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| fixed | 0.9895 | 0.975 | 100% |
| sureshbabu | 0.9538 | 0.935 | 100% |
| forest | 0.9875 | 0.965 | 100% |
| temporal | 0.9999 | 0.999 | 100% |
| temporal+ML | 0.9746 | 0.899 | 100% |
| newton | 1.0000 | 1.000 | 100% |
| newton_reuseH | 1.0000 | 0.997 | 100% |
| select | 1.0000 | 0.997 | 100% |

Figure: `results/fig_cost_stream_main_p1.png`

### p = 2, ALL (160 graphs)

| method | n | median cost (IQR) | mean | median diff vs ref [95% CI] | r_rb | cheaper on | p (Holm) |
|---|---|---|---|---|---|---|---|
| fixed | 160 | 64.0 (48.0-72.0) | 63.1 | reference | | | |
| forest | 160 | 56.0 (32.0-72.0) | 50.7 | -8.0 [-16.0, -8.0] | -0.64 | 65% | 1.3e-10 |
| temporal | 160 | 32.0 (32.0-48.0) | 35.1 | -32.0 [-32.0, -24.0] | -0.91 | 81% | 1.3e-20 |
| temporal+ML | 160 | 8.0 (8.0-32.0) | 21.0 | -40.0 [-40.0, -40.0] | -0.98 | 98% | 3.1e-26 |
| newton | 160 | 80.0 (80.0-80.0) | 83.5 | +16.0 [+16.0, +24.0] | +0.81 | 12% | 1.4e-17 |
| newton_reuseH | 160 | 16.0 (16.0-64.0) | 39.3 | -32.0 [-36.0, -24.0] | -0.69 | 69% | 1.2e-12 |
| select | 160 | 18.0 (18.0-68.0) | 40.1 | -30.0 [-34.0, -22.0] | -0.65 | 71% | 1.5e-12 |

| method | F(start)/F_best median | min | success |
|---|---|---|---|
| fixed | 0.9742 | 0.800 | 100% |
| forest | 0.9851 | 0.837 | 100% |
| temporal | 0.9977 | 0.971 | 100% |
| temporal+ML | 0.9993 | 0.729 | 100% |
| newton | 1.0000 | 0.922 | 100% |
| newton_reuseH | 0.9998 | 0.856 | 100% |
| select | 0.9998 | 0.921 | 100% |

Figure: `results/fig_cost_stream_main_p2.png`

## QAOA vs Goemans-Williamson (approximation ratio = value / MaxCut)

| p | family | graphs | QAOA F_best/MaxCut median | GW expected median | GW best-of-100 median | SDP/MaxCut median |
|---|---|---|---|---|---|---|
| 1 | ALL | 411 | 0.7803 | 0.9621 | 1.0000 | 1.0152 |
| 1 | DVRP | 187 | 0.7495 | 0.9851 | 1.0000 | 1.0002 |
| 1 | ER | 112 | 0.8085 | 0.9595 | 1.0000 | 1.0239 |
| 1 | REG | 112 | 0.7816 | 0.9551 | 1.0000 | 1.0330 |
| 2 | ALL | 411 | 0.8519 | 0.9621 | 1.0000 | 1.0152 |
| 2 | DVRP | 187 | 0.8358 | 0.9851 | 1.0000 | 1.0002 |
| 2 | ER | 112 | 0.8723 | 0.9595 | 1.0000 | 1.0239 |
| 2 | REG | 112 | 0.8567 | 0.9551 | 1.0000 | 1.0330 |

Reading: QAOA's F_best is an EXPECTED cut of the measured state (one shot); GW's expected value is the fair comparison. Both are far from a proof of quantum advantage at these sizes.
