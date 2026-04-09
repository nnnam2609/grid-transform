# Branch Status

`main` is the canonical branch for cleanup, shared helpers, and stable CLI wrappers.

Archive-only branches:

- `Fix/grid-straight-line-without-mid-plate`
- `feat/app-annotation-to-grid-transform`
- `exp/average-speaker-and-speaker-variability`

These branches are retained as historical snapshots only. They should not receive new cleanup work, refactors, or fresh report/export code.

Experimental branch pending triage:

- `exp/compare-males-and-females`

Only reusable fixes should be backported from that branch into `main`. Experimental figures, branch-specific CSV exports, and one-off analysis code stay outside the reusable core.

Current repo policy:

- `grid_transform/` keeps reusable library logic.
- `scripts/run/` stays the canonical CLI surface.
- `tools/` holds report/export and other one-off utilities.
- `experiments/report/` is local-only and should not contain tracked Git files.
