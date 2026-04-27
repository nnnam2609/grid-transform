# Release Checklist

Use this checklist before pushing a public update of the repository.

- Confirm `README.md` renders correctly and all images resolve from `docs/assets/github/`.
- Confirm `outputs/` is not part of the publish set.
- Confirm notebooks open after output stripping and do not contain stale result blobs.
- Search docs and README for machine-specific paths or private infrastructure references.
- Review newly added assets for size and relevance.
- If `VTLN/data/` changed intentionally, rebuild the matching versioned release asset and refresh its `manifest.json` plus `sha256`.
- Verify that every command documented in the README exists under `scripts/run/`.
- Regenerate figures locally if a README asset or report figure was changed intentionally.
- Review the final staged diff once `git` is available on the machine.
