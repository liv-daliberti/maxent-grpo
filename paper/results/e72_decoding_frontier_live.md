# E72 decoding frontier (live)

Cells measured: 300 (stage A target 300).
Reproduction gate: PASS (200/200 checks).
Tolerance: abs(delta) <= max(3.0 * published draw SE, 0.02 for rates / 0.05 for mode counts).

## E2 temperature-repair index

`rho` is the best breadth an arm reaches at ANY temperature, divided by
xGRPO's temperature-one breadth. `rho < 1` means no decoding setting
recovers the treatment's breadth. The accuracy column is what that best
breadth cost.

| domain | arm | best T | distinct@8 at best T | accuracy there | distinct@8 at T=1 | rho |
| --- | --- | --- | --- | --- | --- | --- |
| countdown | drgrpo | 1.6 | 0.650 | 0.523 | 0.627 | 0.34 |
| countdown | xgrpo | 1.3 | 1.923 | 0.582 | 1.893 | 1.02 |
| graph_coloring | drgrpo | 1.3 | 0.327 | 0.322 | 0.325 | 0.14 |
| graph_coloring | xgrpo | 1 | 2.406 | 0.407 | 2.406 | 1.00 |
| mathir | drgrpo | 2 | 0.675 | 0.363 | 0.666 | 0.73 |
| mathir | xgrpo | 1.6 | 0.986 | 0.644 | 0.926 | 1.06 |
| pantry_plan | drgrpo | 2 | 0.719 | 0.522 | 0.634 | 0.33 |
| pantry_plan | xgrpo | 2 | 2.896 | 0.463 | 2.186 | 1.32 |
| python_factors | drgrpo | 0.5 | 0.172 | 0.172 | 0.172 | 0.11 |
| python_factors | xgrpo | 1 | 1.594 | 0.674 | 1.594 | 1.00 |

## E1 frontier points (five-seed means)

| domain | arm | T | mean@8 | pass@8 | distinct@8 | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| countdown | drgrpo | 0.5 | 0.587 | 0.591 | 0.602 | 5 |
| countdown | drgrpo | 0.7 | 0.587 | 0.591 | 0.611 | 5 |
| countdown | drgrpo | 1 | 0.587 | 0.594 | 0.627 | 5 |
| countdown | drgrpo | 1.3 | 0.584 | 0.595 | 0.643 | 5 |
| countdown | drgrpo | 1.6 | 0.523 | 0.595 | 0.650 | 5 |
| countdown | drgrpo | 2 | 0.207 | 0.546 | 0.585 | 5 |
| countdown | xgrpo | 0.5 | 0.622 | 0.667 | 1.606 | 5 |
| countdown | xgrpo | 0.7 | 0.618 | 0.674 | 1.757 | 5 |
| countdown | xgrpo | 1 | 0.610 | 0.693 | 1.893 | 5 |
| countdown | xgrpo | 1.3 | 0.582 | 0.698 | 1.923 | 5 |
| countdown | xgrpo | 1.6 | 0.480 | 0.695 | 1.820 | 5 |
| countdown | xgrpo | 2 | 0.106 | 0.515 | 0.704 | 5 |
| graph_coloring | drgrpo | 0.5 | 0.322 | 0.323 | 0.323 | 5 |
| graph_coloring | drgrpo | 0.7 | 0.322 | 0.325 | 0.325 | 5 |
| graph_coloring | drgrpo | 1 | 0.322 | 0.325 | 0.325 | 5 |
| graph_coloring | drgrpo | 1.3 | 0.322 | 0.327 | 0.327 | 5 |
| graph_coloring | drgrpo | 1.6 | 0.323 | 0.327 | 0.327 | 5 |
| graph_coloring | drgrpo | 2 | 0.158 | 0.326 | 0.326 | 5 |
| graph_coloring | xgrpo | 0.5 | 0.449 | 0.889 | 1.992 | 5 |
| graph_coloring | xgrpo | 0.7 | 0.431 | 0.934 | 2.263 | 5 |
| graph_coloring | xgrpo | 1 | 0.407 | 0.961 | 2.406 | 5 |
| graph_coloring | xgrpo | 1.3 | 0.381 | 0.965 | 2.404 | 5 |
| graph_coloring | xgrpo | 1.6 | 0.298 | 0.929 | 2.011 | 5 |
| graph_coloring | xgrpo | 2 | 0.056 | 0.364 | 0.439 | 5 |
| mathir | drgrpo | 0.5 | 0.645 | 0.654 | 0.654 | 5 |
| mathir | drgrpo | 0.7 | 0.644 | 0.659 | 0.659 | 5 |
| mathir | drgrpo | 1 | 0.644 | 0.666 | 0.666 | 5 |
| mathir | drgrpo | 1.3 | 0.641 | 0.669 | 0.671 | 5 |
| mathir | drgrpo | 1.6 | 0.601 | 0.670 | 0.672 | 5 |
| mathir | drgrpo | 2 | 0.363 | 0.675 | 0.675 | 5 |
| mathir | xgrpo | 0.5 | 0.755 | 0.827 | 0.852 | 5 |
| mathir | xgrpo | 0.7 | 0.746 | 0.840 | 0.886 | 5 |
| mathir | xgrpo | 1 | 0.735 | 0.867 | 0.926 | 5 |
| mathir | xgrpo | 1.3 | 0.722 | 0.887 | 0.964 | 5 |
| mathir | xgrpo | 1.6 | 0.644 | 0.893 | 0.986 | 5 |
| mathir | xgrpo | 2 | 0.369 | 0.875 | 0.954 | 5 |
| pantry_plan | drgrpo | 0.5 | 0.524 | 0.545 | 0.597 | 5 |
| pantry_plan | drgrpo | 0.7 | 0.524 | 0.552 | 0.623 | 5 |
| pantry_plan | drgrpo | 1 | 0.524 | 0.556 | 0.634 | 5 |
| pantry_plan | drgrpo | 1.3 | 0.523 | 0.558 | 0.649 | 5 |
| pantry_plan | drgrpo | 1.6 | 0.522 | 0.564 | 0.677 | 5 |
| pantry_plan | drgrpo | 2 | 0.522 | 0.575 | 0.719 | 5 |
| pantry_plan | xgrpo | 0.5 | 0.604 | 0.767 | 1.474 | 5 |
| pantry_plan | xgrpo | 0.7 | 0.593 | 0.810 | 1.726 | 5 |
| pantry_plan | xgrpo | 1 | 0.566 | 0.869 | 2.186 | 5 |
| pantry_plan | xgrpo | 1.3 | 0.527 | 0.911 | 2.488 | 5 |
| pantry_plan | xgrpo | 1.6 | 0.500 | 0.937 | 2.680 | 5 |
| pantry_plan | xgrpo | 2 | 0.463 | 0.952 | 2.896 | 5 |
| python_factors | drgrpo | 0.5 | 0.172 | 0.172 | 0.172 | 5 |
| python_factors | drgrpo | 0.7 | 0.172 | 0.172 | 0.172 | 5 |
| python_factors | drgrpo | 1 | 0.172 | 0.172 | 0.172 | 5 |
| python_factors | drgrpo | 1.3 | 0.172 | 0.172 | 0.172 | 5 |
| python_factors | drgrpo | 1.6 | 0.119 | 0.172 | 0.172 | 5 |
| python_factors | drgrpo | 2 | 0.029 | 0.143 | 0.143 | 5 |
| python_factors | xgrpo | 0.5 | 0.681 | 0.688 | 1.380 | 5 |
| python_factors | xgrpo | 0.7 | 0.681 | 0.688 | 1.500 | 5 |
| python_factors | xgrpo | 1 | 0.674 | 0.688 | 1.594 | 5 |
| python_factors | xgrpo | 1.3 | 0.631 | 0.688 | 1.538 | 5 |
| python_factors | xgrpo | 1.6 | 0.374 | 0.687 | 1.210 | 5 |
| python_factors | xgrpo | 2 | 0.038 | 0.136 | 0.136 | 5 |
