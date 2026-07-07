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
- Typed model-selection errors (`ModelNotFoundError`, `GatedModelError`,
  `RemoteCodeRequiredError`, `UnsupportedArchitectureError`,
  `ModelEnvironmentError`, `ESMCForkRequiredError`) with actionable messages
  when HuggingFace config or model loading fails.
- Gated integration test for the generic HuggingFace embedder path
  (`GENERIC_HF_TEST=1`; downloads `hf-internal-testing/tiny-random-BertModel`).
  CI runs it in the integration job (model-download) while the unit job stays
  download-free.
- `--trust_remote_code` CLI/API flag to opt in to HuggingFace custom modeling code
  (default off; documented security risk in `--help`).
- `pepe --check_model <repo>` dry-run that loads config and tokenizer only and
  prints architecture, embedder choice, max length, tokenizer type, and output
  capabilities before any embedding run.
- `--verbose` CLI/API flag to enable GPU/IO profiling stats at the end of an
  embedding run (default off).
- Streaming vs in-memory regression tests (`test_streaming_roundtrip.py`)
  covering all standard output types.
- Session-scoped pytest fixture that loads ESM2-8M once per test session;
  `@pytest.mark.integration` and `@pytest.mark.slow` markers registered in
  `conftest.py`.

### Changed
- Logger namespace unified under `pepe` (was `src.*`) so API and module loggers
  share handlers.
- HuggingFace dataset tokenization batches all sequences in one tokenizer call
  instead of a per-sequence loop.
- Per-batch `gc.collect()` and non-OOM `torch.cuda.empty_cache()` removed from
  the embed loop (CUDA cache is still cleared on OOM retry paths).
- GPU/IO profiling summary is logged only when `--verbose` is set.
- HuggingFace config/load failures are classified by upstream exception type
  instead of string-sniffing on error messages (`model_errors.py`).
- Generic HuggingFace model loading no longer auto-enables `trust_remote_code`
  based on repo name patterns; use `--trust_remote_code` explicitly when needed.

### Fixed
- Streaming mode silently dropped attention outputs due to memmap registry key
  mismatches (`attention_matrices_*` vs `attention_head` / `attention_layer` /
  `attention_model`).
- `preallocate_disk_space()` failed for model-level outputs such as
  `attention_model` when assigning memmaps via `setattr` on a dict.
- `get_substring_positions` did not raise when a CSV substring entry was missing
  for a sequence label.
- Substring pooling produced NaN embeddings when a substring did not match the
  tokenized full sequence; now raises `ValueError` with the sequence label.
- Missing substring CSV entries raised a generic `assert` (disabled under
  `python -O`); replaced with `ValueError` listing missing IDs.
- `MultiIODispatcher` default `heavy_output_type` updated from legacy
  `embeddings_unpooled` to `per_token`.
- Generic HuggingFace embedder (`GenericHuggingFaceEmbedder`) now mirrors ESM-2
  safeguards: length-limit detection and optional sequence splitting via
  `_check_max_input_length`, eager attention when extracting contacts, and a
  warning when the tokenizer is subword-based (per-residue outputs may be
  misaligned). Sentinel `tokenizer.model_max_length` values (~1e30) are treated
  as unknown limits instead of a real cap.

## [1.3.0] - 2026-07-07

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
