# Changelog

All notable changes to PEPE are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
given a version `MAJOR.MINOR.PATCH`, increment

- **MAJOR** when you make an incompatible change (e.g. rename/remove a CLI flag,
  change `embed()`'s signature, or change the output file format),
- **MINOR** when you add functionality in a backward-compatible way,
- **PATCH** when you make backward-compatible bug fixes.

When you cut a release, move items from `Unreleased` into a new dated version
section and bump `__version__` in `src/pepe/__init__.py` (the single source of
truth that drives publishing).

## [Unreleased]

### Added
- Broader model compatibility: BERT-like and other unrecognized HuggingFace
  architectures now fall back to the generic `AutoModel`-based embedder instead
  of raising, so standard protein encoders such as ProtBert, AntiBERTy and
  IgBert work from a repo id with no code changes.
- Continuous-integration workflow (`.github/workflows/test.yml`) that runs the
  test suite on every push and pull request.
- `CONTRIBUTING.md` with dev-environment setup, how to run tests, and the
  release process.
- This changelog.
- `rjieba` as a runtime dependency, required by AntiBERTa2's `RoFormerTokenizer`.
  Previously AntiBERTa2 models failed with an `ImportError` unless users
  installed it manually; CI now guards against this regression.

### Changed
- Model dispatch (`model_selecter.py`) now inspects a HuggingFace repo's config
  (`AutoConfig.model_type`) before matching on its name. A fine-tune whose repo
  slug contains "esm2" but whose architecture is something else is no longer
  mis-routed. Bare ESM weight names (e.g. `esm2_t6_8M_UR50D`) and local
  checkpoint paths keep their existing fast, no-download handling.
- Bumped all Node-based GitHub Actions to their Node 24 majors across the test
  and publish workflows (`actions/checkout` v4→v5, `actions/setup-python`
  v4/v5→v6, `softprops/action-gh-release` v1→v3, `conda-incubator/setup-miniconda`
  v3→v4), ahead of GitHub removing the Node 20 runtime in fall 2026.
  `pypa/gh-action-pypi-publish` is Docker-based and unaffected.

### Fixed
- Corrected the PyPI license classifier from "GNU Affero General Public License
  v3" to "MIT License" in `pyproject.toml` and `setup.py`, matching the actual
  `LICENSE` file.

### Removed
- Stray nested clones of the repository (`pepe-cli/`, `pepe-cli-1/`) that were
  sitting inside the working tree.

## [1.2.1]

### Fixed
- TestPyPI publishing: resolved version-collision and packaging-metadata issues
  on test-branch pushes.

## [1.2.0]

### Added
- ESMC model support (`biohub/ESMC-*`), loaded via Biohub's `transformers` fork.

### Changed
- ESM-2 models are now loaded through HuggingFace `transformers` instead of
  `fair-esm`.

## [1.1.1]

### Fixed
- Removed a duplicate default on the `--device` argument.

---

*Versions prior to 1.1.1 predate this changelog; see the git history for details.*
