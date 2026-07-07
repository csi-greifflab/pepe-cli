# Bioconda Submission Guide for PEPE

This guide outlines the steps required to submit `pepe-cli` to the [Bioconda](https://bioconda.github.io/) channel. Bioconda uses a recipe-based system where you submit a Pull Request to their repository.

## Prerequisites

1.  **Stable Release**: Ensure you have published a stable version of `pepe-cli` to [PyPI](https://pypi.org/). Bioconda usually avoids development/pre-release versions.
2.  **GitHub Account**: You need to be able to fork the `bioconda-recipes` repository.

## Bioconda-Compatible Recipe

Save the following content as `meta.yaml` when creating your PR to Bioconda.

```yaml
{% set name = "pepe-cli" %}
{% set version = "1.0.4" %} # Update to your latest stable version

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/pepe_cli-{{ version }}.tar.gz
  sha256: YOUR_SHA256_HASH_HERE # Get this from PyPI or use: openssl sha256 <file>

build:
  number: 0
  noarch: python
  script: {{ PYTHON }} -m pip install . -vv
  run_exports:
    - {{ pin_subpackage(name|lower, max_pin="x.x") }}
  entry_points:
    - pepe = pepe.__main__:main
    - pepe-cli = pepe.__main__:main

requirements:
  host:
    - python >=3.8
    - pip
  run:
    - python >=3.8
    - pytorch >=1.9.0
    - transformers >=4.20.0
    - sentencepiece
    - numpy
    - protobuf
    - alive-progress

test:
  imports:
    - pepe
  commands:
    - pepe --help

about:
  home: https://github.com/csi-greifflab/pepe-cli
  license: GNU Affero General Public License v3.0 or later
  license_family: AGPL
  license_file: LICENSE
  summary: Pipeline for Easy Protein Embedding
  description: |
    PEPE (Pipeline for Easy Protein Embedding) is a tool for extracting
    embeddings and attention matrices from protein sequences using pre-trained models.
  doc_url: https://github.com/csi-greifflab/pepe-cli#readme
  dev_url: https://github.com/csi-greifflab/pepe-cli

extra:
  recipe-maintainers:
    - your-github-handle
```

## Step-by-Step Submission

1.  **Fork Bioconda Recipes**:
    - Go to [bioconda/bioconda-recipes](https://github.com/bioconda/bioconda-recipes) and fork it.
2.  **Clone your fork**:
    ```bash
    git clone https://github.com/your-username/bioconda-recipes.git
    cd bioconda-recipes
    ```
3.  **Create a new branch**:
    ```bash
    git checkout -b add-pepe-cli
    ```
4.  **Add the recipe**:
    - Create a directory: `recipes/pepe-cli/`
    - Create `recipes/pepe-cli/meta.yaml` with the content provided above.
5.  **Test the recipe locally** (optional but recommended):
    - Requires `conda-build` and `anaconda-client`.
    ```bash
    conda build recipes/pepe-cli/
    ```
6.  **Commit and Push**:
    ```bash
    git add recipes/pepe-cli/meta.yaml
    git commit -m "Add pepe-cli recipe"
    git push origin add-pepe-cli
    ```
7.  **Create a Pull Request**:
    - Go to the Bioconda repository and create a PR from your branch.
    - Follow the PR template instructions.
    - Bioconda's CI will automatically build and test your recipe.
8.  **Wait for Review**:
    - Bioconda maintainers will review your PR. Address any feedback they provide.

Once merged, your package will be available via:
```bash
conda install -c bioconda pepe-cli
```
