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
- METL 1D protein embeddings via optional `metl-pretrained` backend: install with
  `pip install pepe-cli[metl]` and use model identifiers such as `metl-g-20m-1d`
  (and other `metl-*-1d` names). Dispatch lives in `model_selecter.py`; embedding
  is handled by `METLEmbedder` with `METLDataset` tokenization.
- `[metl]` and `[esm]` optional dependency extras in `pyproject.toml` and
  `setup.py`.
- Typed errors `METLPackageRequiredError` and `METL3DNotSupportedError` when
  METL is requested without the extra or when a 3D METL model id is used.
- Unit tests for METL dispatch and an optional integration test (`METL_TEST=1`)
  wired in CI (`.github/workflows/test.yml`).

### Limitations
- METL support is **1D embeddings only** (`per_token`, `mean_pooled`,
  `substring_pooled`). Logits and attention outputs are not available. 3D METL
  model identifiers are rejected with a clear error.

### Removed
- Dropped support for Python 3.8 and 3.9. The minimum supported version is now
  **3.10** (`requires-python >=3.10`), matching the mypy/Ruff target. The CI unit
  matrix, packaging classifiers, and the conda recipe were updated accordingly.

## [1.4.0] - 2026-07-07

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

- PEP 561 typing support via `src/pepe/py.typed` and setuptools package data.
- `[tool.mypy]` configuration plus a non-blocking CI job that type-checks the
  public API surface (`api`, model selection, embedders, `utils`).
- Unit test guarding CLI/`embed()` argument parity (`test_sync_arguments.py`).
- Custom `.pt` embedder round-trip test (`test_custom_embedder.py`).
- Gated ESM-1 integration test (`test_esm1_integration.py`; requires `fair-esm`).
- Gated T5 / AntiBERTa2 integration tests (`T5_ANTIBERTA2_TEST=1`); optional
  manual ProtT5 checkpoint test (`PROT_T5_MANUAL_TEST=1`).
- Regression test for mean-pooled reconstruction of split sequences without
  per_token (`test_reconstruct_mean_pooled.py`).

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

- Type annotations on the embedding engine and `check_untyped_defs` mypy overrides
  for core modules (`base_embedder`, HuggingFace/custom/ESM embedders, `utils`,
  `model_selecter`).
- CI unit job reports pytest coverage (`--cov=pepe`) and runs sync/custom
  embedder unit tests.
- Legacy manual verification scripts removed in favor of pytest
  (`test_run.py`, `verify_readme.py`, `verify_cross_tool_consistency.py`).

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

- In-memory long-sequence chunk reconstruction keyed outputs by output type
  instead of Python object id, fixing mean-pooled (and related) stitch-up after
  splits.
- In-memory long-sequence reconstruction no longer crashes with `KeyError:
  'per_token'` when `mean_pooled` is requested without `per_token`; per-token
  representations are now retained internally so the pooled vector can be
  stitched back from chunks (and are not exported).
- Custom embedder checkpoint load uses explicit `weights_only=False` for PyTorch
  2.x `.pt` checkpoints that include non-tensor metadata.
- Custom embedder `_infer_num_heads` returns a default when inference fails
  instead of implicitly returning `None`.
- mypy CI failure for `T5TokenizerFast` lazy import in `huggingface_embedder.py`
  (transformers stubs omit the top-level re-export; import via submodule with
  transformers 5.x fallback).
- `T5Embedder` failed to load production ProtT5 checkpoints
  (`Rostlab/prot_t5_xl_half_uniref50-enc`) after the switch to `T5TokenizerFast`
  — their SentencePiece model cannot be converted to a fast tokenizer
  (`Unigram ... trained with a different algorithm`). Tokenizer loading now falls
  back to the slow `T5Tokenizer` when the fast conversion fails, so ProtT5 works
  again while fast-only Hub checkpoints keep using `T5TokenizerFast`.

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
