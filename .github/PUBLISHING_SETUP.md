# GitHub Actions Setup for PyPI Publishing

This repository contains GitHub Actions workflows for automatically building and publishing the `pypepe` package to PyPI and TestPyPI.

## Workflows

### 1. `publish-test-branch.yml` - API Token Based (Recommended for quick setup)
- **Triggers**: Pushes to `test` branch, manual workflow dispatch
- **Purpose**: Publishes test versions to TestPyPI
- **Authentication**: Uses API tokens stored in GitHub Secrets

### 2. `publish-test-branch-trusted.yml` - Trusted Publishing (More secure)
- **Triggers**: Pushes to `test` branch, manual workflow dispatch
- **Purpose**: Publishes test versions to TestPyPI using trusted publishing
- **Authentication**: Uses OpenID Connect (no API tokens needed)

### 3. `publish.yml` - Main Publishing Workflow
- **Triggers**: Pushes to main/master, test tags, releases
- **Purpose**: Publishes to TestPyPI (test versions) and PyPI (releases)

### 4. `publish-trusted.yml` - Trusted Publishing for Releases
- **Triggers**: Published releases
- **Purpose**: Publishes to PyPI using trusted publishing

## Setup Instructions

### Option 1: Using API Tokens (Easier setup)

1. **Get TestPyPI API Token**:
   - Go to [TestPyPI](https://test.pypi.org/account/login/)
   - Log in or create an account
   - Go to Account Settings → API Tokens
   - Create a new token with scope for the entire account or specific project
   - Copy the token (it starts with `pypi-`)

2. **Add Token to GitHub Secrets**:
   - Go to your GitHub repository
   - Navigate to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: Paste your TestPyPI API token
   - Click "Add secret"

3. **Enable the workflow**:
   - The `publish-test-branch.yml` workflow will automatically run when you push to the `test` branch

### Option 2: Using Trusted Publishing (More secure, recommended)

1. **Configure Trusted Publishing on TestPyPI**:
   - Go to [TestPyPI](https://test.pypi.org/account/login/)
   - Log in to your account
   - Go to Account Settings → Publishing
   - Add a new trusted publisher with:
     - Owner: `your-github-username`
     - Repository: `pypepe`
     - Workflow: `publish-test-branch-trusted.yml`
     - Environment: (leave empty for now)

2. **Enable the workflow**:
   - The `publish-test-branch-trusted.yml` workflow will automatically run when you push to the `test` branch

## Usage

### Publishing Test Versions

1. **Automatic**: Push commits to the `test` branch:
   ```bash
   git checkout test
   git add .
   git commit -m "Test changes"
   git push origin test
   ```

2. **Manual**: Trigger the workflow manually:
   - Go to Actions tab in your GitHub repository
   - Select the workflow you want to run
   - Click "Run workflow"
   - Choose the branch and click "Run workflow"

### Version Management

The test branch workflows automatically update the version number to include:
- Timestamp (YYYYMMDDHHMMSS)
- Git commit hash (short)
- Test identifier

Example: `1.0.0.dev20250714123456+abc1234`

### Installing Test Versions

After publishing to TestPyPI, install with:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pypepe
```

## Troubleshooting

### Common Issues

1. **"Invalid or non-existent authentication information"**:
   - Check that your API token is correctly set in GitHub Secrets
   - Ensure the token has the correct permissions

2. **"File already exists"**:
   - TestPyPI doesn't allow overwriting existing versions
   - The workflow automatically generates unique versions to avoid this

3. **"Package name already exists"**:
   - Someone else might have claimed the package name
   - Consider using a different package name or contact TestPyPI support

### Monitoring

- Check the Actions tab in your GitHub repository for workflow runs
- Each workflow provides detailed logs for debugging
- Failed runs will show error messages in the logs

## Security Notes

- **API Tokens**: Never commit API tokens to your repository
- **Trusted Publishing**: Preferred method as it doesn't require storing secrets
- **Branch Protection**: Consider protecting your `test` branch to prevent accidental pushes

## Files Modified

This setup creates/modifies:
- `.github/workflows/publish-test-branch.yml`
- `.github/workflows/publish-test-branch-trusted.yml`
- `pyproject.toml` version is automatically updated during workflow runs

## Conda Publishing Setup

The repository is configured to build and publish Conda packages automatically when pushing to the `main` branch.

### Prerequisites

1.  **Anaconda/Miniconda**: Ensure you have Conda installed locally for testing builds.
2.  **Anaconda.org Account**: You need an account on [Anaconda.org](https://anaconda.org/).

### GitHub Configuration

1.  **Get Anaconda API Token**:
    - Log in to your Anaconda.org account.
    - Go to **Settings** → **Access** → **Tokens**.
    - Click **Create new token**.
    - Give it a name (e.g., `GitHub Actions`) and ensure it has `allow uploads to standard channels` permission.
    - Copy the token.

2.  **Add Token to GitHub Secrets**:
    - In your GitHub repository, go to **Settings** → **Secrets and variables** → **Actions**.
    - Click **New repository secret**.
    - Name: `ANACONDA_TOKEN`
    - Value: Paste your Anaconda API token.

### Workflow Integration

The `publish-main-branch-trusted.yml` workflow includes a `publish-conda` job that:
- Sets up a Conda environment.
- Builds the package using the recipe in `.github/conda/meta.yaml`.
- Uploads the package to your Anaconda.org channel with the `main` label.

To verify the setup, you can manually trigger the workflow from the **Actions** tab.
