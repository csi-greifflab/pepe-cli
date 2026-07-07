# Contributing to PEPE

Notes for maintaining and extending PEPE — written to be useful whether you're a
first-time contributor or future-you coming back after six months.

## Development setup

```sh
# Clone, then from the repo root:
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # editable install: code changes take effect immediately
pip install pytest biopython     # test-only dependencies

# Optional backends, only if you work on those models:
pip install fair-esm                                             # ESM-1 models
pip install git+https://github.com/Biohub/transformers.git@main  # ESMC models
```

## Running tests

Always run from the **repository root** (tests use paths like
`src/tests/test_files/`).

```sh
# Full suite
python -m pytest src/tests/

# Fast tests only — mocked, no model downloads (what CI gates every push on)
python -m pytest src/tests/test_parse_arguments.py \
                 src/tests/test_device_logic.py \
                 src/tests/test_load_layers_default.py

# A single test
python -m pytest src/tests/test_api_unittest.py::TestPepeAPI::test_embed_to_disk
```

Some tests download a small model (`esm2_t6_8M_UR50D`, ~30 MB) on first run.
ESMC tests skip themselves automatically if the Biohub fork isn't installed.

**Rule of thumb:** every time you fix a bug, add a test that would have caught
it. That is how the suite becomes a map of everything that has ever gone wrong.

## Branching and pull requests

- `main` is the release branch (publishes to **PyPI**).
- `test` is the integration branch (publishes to **TestPyPI**) and is where
  day-to-day work lands.
- Do work on a short-lived branch off `test`, open a pull request into `test`,
  and let CI go green before merging. Merge `test` into `main` when you're ready
  to cut a public release.
- Don't commit directly to `main` or `test`.

## Making a release

1. Decide the new version number using [SemVer](https://semver.org) (see
   `CHANGELOG.md` for what MAJOR/MINOR/PATCH mean here).
2. Update the `Unreleased` section of `CHANGELOG.md` into a new version section.
3. Bump `__version__` in `src/pepe/__init__.py` — this single value drives all
   packaging metadata (`pyproject.toml` and `setup.py` read from it) and
   triggers the publish workflows. The version on `main` must **not** contain a
   `-dev` or `-test` suffix.
4. Merge to the appropriate branch. The `publish-*.yml` GitHub Actions build and
   upload the package.
5. Tag the release so results stay reproducible: `git tag v<version> && git push --tags`.

## Adding support for a new model

The architecture is documented in `CLAUDE.md`. In short:

- `select_model()` in `src/pepe/model_selecter.py` maps a model name to an
  embedder class — add your dispatch rule there.
- Embedders subclass `BaseEmbedder` (`src/pepe/embedders/base_embedder.py`) and
  typically only need to implement model/tokenizer loading and `_compute_outputs`.
- Keep heavy imports (torch, transformers, esm) **lazy** — imported inside
  functions/methods, not at module top level — so `--help` stays fast and
  optional backends remain optional.
- Adding a new output type means adding both an entry in `_set_output_objects()`
  and the matching `_extract_*` method.

## Deprecating things

You have real users now. When you must change a CLI flag or a public API, keep
the old behavior working with a warning for a release or two rather than
removing it outright.
