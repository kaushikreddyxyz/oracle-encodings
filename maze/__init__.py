"""Maze-transformer injection research repo.

Layout:
  core/         shared library (injection, data, evals, checkpoints, reliance)
  infra/        pod provisioning + launch scripts
  legacy/       original TransformerLens trainers (kept runnable for provenance)
  experiments/  one folder per campaign (scripts / plots / results / figures)

See MIGRATION_SPEC.md for the full map and migration_changes.md for the
consolidation log.
"""
