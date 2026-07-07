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
- Continuous-integration workflow (`.github/workflows/test.yml`) that runs the
  test suite on every push and pull request.
- `CONTRIBUTING.md` with dev-environment setup, how to run tests, and the
  release process.
- This changelog.
- `rjieba` as a runtime dependency, required by AntiBERTa2's `RoFormerTokenizer`.
  Previously AntiBERTa2 models failed with an `ImportError` unless users
  installed it manually; CI now guards against this regression.

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
