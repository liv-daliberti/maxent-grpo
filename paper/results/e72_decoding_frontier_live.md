# E72 decoding frontier (live)

Cells measured: 300 (stage A target 300).
Reproduction gate: FAIL (197/200 checks).
Tolerance: abs(delta) <= max(3.0 * published draw SE, 0.02 for rates / 0.05 for mode counts).

## E2 temperature-repair index

`rho` is the best breadth an arm reaches at ANY temperature, divided by
xGRPO's temperature-one breadth. `rho < 1` means no decoding setting
recovers the treatment's breadth. The accuracy column is what that best
breadth cost.

| domain | arm | best T | distinct@8 at best T | accuracy there | distinct@8 at T=1 | rho |
| --- | --- | --- | --- | --- | --- | --- |
| countdown | control | 1.6 | 0.501 | 0.390 | 0.487 | n/a |
| countdown | replay | 1.3 | 1.670 | 0.569 | 1.632 | n/a |
| graph_coloring | control | 1.6 | 0.350 | 0.301 | 0.325 | n/a |
| graph_coloring | replay | 1 | 2.434 | 0.403 | 2.434 | n/a |
| mathir | control | 1.6 | 0.532 | 0.437 | 0.515 | n/a |
| mathir | replay | 1.6 | 0.876 | 0.636 | 0.840 | n/a |
| pantry_plan | control | 2 | 0.534 | 0.522 | 0.522 | n/a |
| pantry_plan | replay | 2 | 2.573 | 0.478 | 1.535 | n/a |
| python_factors | control | 0.5 | 0.172 | 0.172 | 0.172 | n/a |
| python_factors | replay | 1.6 | 0.576 | 0.286 | 0.571 | n/a |

## E1 frontier points (five-seed means)

| domain | arm | T | mean@8 | pass@8 | distinct@8 | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| countdown | control | 0.5 | 0.465 | 0.472 | 0.473 | 5 |
| countdown | control | 0.7 | 0.464 | 0.475 | 0.477 | 5 |
| countdown | control | 1 | 0.464 | 0.480 | 0.487 | 5 |
| countdown | control | 1.3 | 0.463 | 0.484 | 0.492 | 5 |
| countdown | control | 1.6 | 0.390 | 0.483 | 0.501 | 5 |
| countdown | control | 2 | 0.143 | 0.475 | 0.479 | 5 |
| countdown | replay | 0.5 | 0.622 | 0.661 | 1.384 | 5 |
| countdown | replay | 0.7 | 0.614 | 0.666 | 1.529 | 5 |
| countdown | replay | 1 | 0.595 | 0.672 | 1.632 | 5 |
| countdown | replay | 1.3 | 0.569 | 0.674 | 1.670 | 5 |
| countdown | replay | 1.6 | 0.440 | 0.674 | 1.584 | 5 |
| countdown | replay | 2 | 0.103 | 0.463 | 0.638 | 5 |
| graph_coloring | control | 0.5 | 0.323 | 0.323 | 0.323 | 5 |
| graph_coloring | control | 0.7 | 0.323 | 0.323 | 0.323 | 5 |
| graph_coloring | control | 1 | 0.323 | 0.323 | 0.325 | 5 |
| graph_coloring | control | 1.3 | 0.323 | 0.330 | 0.333 | 5 |
| graph_coloring | control | 1.6 | 0.301 | 0.341 | 0.350 | 5 |
| graph_coloring | control | 2 | 0.118 | 0.307 | 0.318 | 5 |
| graph_coloring | replay | 0.5 | 0.468 | 0.943 | 2.172 | 5 |
| graph_coloring | replay | 0.7 | 0.444 | 0.963 | 2.395 | 5 |
| graph_coloring | replay | 1 | 0.403 | 0.968 | 2.434 | 5 |
| graph_coloring | replay | 1.3 | 0.373 | 0.963 | 2.368 | 5 |
| graph_coloring | replay | 1.6 | 0.236 | 0.880 | 1.673 | 5 |
| graph_coloring | replay | 2 | 0.034 | 0.254 | 0.266 | 5 |
| mathir | control | 0.5 | 0.466 | 0.491 | 0.493 | 5 |
| mathir | control | 0.7 | 0.464 | 0.500 | 0.502 | 5 |
| mathir | control | 1 | 0.462 | 0.513 | 0.515 | 5 |
| mathir | control | 1.3 | 0.459 | 0.521 | 0.523 | 5 |
| mathir | control | 1.6 | 0.437 | 0.530 | 0.532 | 5 |
| mathir | control | 2 | 0.204 | 0.495 | 0.496 | 5 |
| mathir | replay | 0.5 | 0.704 | 0.764 | 0.783 | 5 |
| mathir | replay | 0.7 | 0.699 | 0.778 | 0.802 | 5 |
| mathir | replay | 1 | 0.690 | 0.803 | 0.840 | 5 |
| mathir | replay | 1.3 | 0.678 | 0.822 | 0.866 | 5 |
| mathir | replay | 1.6 | 0.636 | 0.823 | 0.876 | 5 |
| mathir | replay | 2 | 0.331 | 0.779 | 0.809 | 5 |
| pantry_plan | control | 0.5 | 0.522 | 0.522 | 0.522 | 5 |
| pantry_plan | control | 0.7 | 0.522 | 0.522 | 0.522 | 5 |
| pantry_plan | control | 1 | 0.522 | 0.522 | 0.522 | 5 |
| pantry_plan | control | 1.3 | 0.522 | 0.522 | 0.522 | 5 |
| pantry_plan | control | 1.6 | 0.522 | 0.522 | 0.522 | 5 |
| pantry_plan | control | 2 | 0.522 | 0.525 | 0.534 | 5 |
| pantry_plan | replay | 0.5 | 0.566 | 0.655 | 1.085 | 5 |
| pantry_plan | replay | 0.7 | 0.560 | 0.675 | 1.209 | 5 |
| pantry_plan | replay | 1 | 0.542 | 0.730 | 1.535 | 5 |
| pantry_plan | replay | 1.3 | 0.531 | 0.795 | 1.901 | 5 |
| pantry_plan | replay | 1.6 | 0.511 | 0.836 | 2.239 | 5 |
| pantry_plan | replay | 2 | 0.478 | 0.870 | 2.573 | 5 |
| python_factors | control | 0.5 | 0.172 | 0.172 | 0.172 | 5 |
| python_factors | control | 0.7 | 0.172 | 0.172 | 0.172 | 5 |
| python_factors | control | 1 | 0.172 | 0.172 | 0.172 | 5 |
| python_factors | control | 1.3 | 0.172 | 0.172 | 0.172 | 5 |
| python_factors | control | 1.6 | 0.150 | 0.172 | 0.172 | 5 |
| python_factors | control | 2 | 0.020 | 0.163 | 0.163 | 5 |
| python_factors | replay | 0.5 | 0.518 | 0.525 | 0.563 | 5 |
| python_factors | replay | 0.7 | 0.515 | 0.525 | 0.563 | 5 |
| python_factors | replay | 1 | 0.507 | 0.528 | 0.571 | 5 |
| python_factors | replay | 1.3 | 0.442 | 0.527 | 0.568 | 5 |
| python_factors | replay | 1.6 | 0.286 | 0.524 | 0.576 | 5 |
| python_factors | replay | 2 | 0.026 | 0.170 | 0.170 | 5 |
