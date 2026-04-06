
# isoperimetric-profile

Compute the exact **relative isoperimetric profile**

$$
I_P(A) = \inf\{\mathrm{Per}(\Omega;P) : \Omega\subset P,\ |\Omega| = A\}
$$

for one strictly convex polygon $P \subset \mathbb{R}^2$.

The package precomputes exact branch families for admissible minimizers:
- circular-arc branches (disk-type competitors),
- strip branches (parallel-edge competitors),

then evaluates $I_P(A)$ exactly on requested areas by taking the lower envelope of those branches.

It also computes:
- the disk benchmark profile $I_D(A)$ for the disk with the same total area,
- the ratio $I_P(A) / I_D(A)$,
- a coarse/fine numerical search of the maximal ratio area.

## Install

Core package:

```bash
pip install .
```

Plotting support (optional):

```bash
pip install ".[plot]"
```

## Python API

```python
from isoperimetric_profile import (
    precompute_profile,
    profile_on_uniform_grid,
    plot_isoperimetric_profile,
    profile_on_areas,
    candidate_at_area,
    max_relative_ratio,
)

polygon = [(0.0, 0.0), (2.0, 0.0), (2.5, 1.0), (1.2, 2.0), (0.0, 1.0)]

pre = precompute_profile(polygon)
grid = profile_on_uniform_grid(pre, num_points=256)

# Saves I_P(A) sampled on the grid
plot_isoperimetric_profile(grid, path="profile.png")

# Coarse/fine search for max_A I_P(A)/I_D(A)
best = max_relative_ratio(pre)

print(best.best_ratio, best.best_area)
```

## CLI

Input polygon files can be JSON

```json
[[0, 0], [2, 0], [2.5, 1], [1.2, 2], [0, 1]]
```

or plain text

```text
0 0
2 0
2.5 1
1.2 2
0 1
```

Basic run:

```bash
python3 -m isoperimetric_profile polygon.txt --output result.json
```

Query a specific area (includes exact minimizing candidate metadata in output):

```bash
python3 -m isoperimetric_profile polygon.txt --query-area 0.3 --output result.json
```

Export sampled profile grid to CSV:

```bash
python3 -m isoperimetric_profile polygon.txt --profile-csv profile.csv
```

Plot sampled profile $I_P(A)$:

```bash
python3 -m isoperimetric_profile polygon.txt --plot profile.png
```

Common options:
- `--num-points`: number of sampled areas on $(\varepsilon, |P|/2 - \varepsilon)$.
- `--area-epsilon`: endpoint cutoff for the sampling interval.
- `--center`: center polygon at centroid before evaluation.
- `--normalize-area X`: rescale and center polygon to total area `X`.
- `--coarse-points`, `--fine-points`, `--local-window`: controls the coarse/fine max-ratio search.
- `--plot-show`: show an interactive plot window.
- `--plot-title`: custom plot title.

## Outputs

CLI JSON includes:
- cleaned polygon and total area,
- sampled profile arrays (`query_areas`, `profile`, `disk_profile`, `ratio`, branch family),
- maximum sampled ratio summary,
- optional single-area query payload with minimizing candidate data.

## Assumptions and Scope

- Input polygon must be strictly convex (CCW orientation is enforced internally).
- Degenerate or non-convex polygons are rejected.
- The maximizing area from `max_relative_ratio` is a numerical search over the exact evaluator (coarse grid + local refinement), not a symbolic global maximization.
