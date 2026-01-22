# Release Build Scripts

These scripts are used by both the GitHub Actions workflow and for local testing. They allow you to test the manylinux package building process locally before pushing to GitHub Actions.

## Prerequisites

- Docker installed and running
- Python 3.11 (for preparing common files)
- Bash shell

## Usage

### Quick Start

Build the manylinux package with a single command:

```bash
./scripts/build-all.sh [VERSION]
```

If no version is specified, it defaults to `dev`.

### Step-by-Step

#### 1. Prepare Common Files

This step prepares the shared files and downloads the retrieval models:

```bash
./scripts/prepare-common.sh [VERSION]
```

This creates a `release_common/` directory with:
- All Python source files
- Requirements and constraints
- Downloaded models
- Configuration files (Continue, Roo Code, Cline + rules)

#### 2. Build Package

Build the manylinux_2_34 package:

```bash
./scripts/package-manylinux-2_34.sh [VERSION]
```

The script will:
- Run in a Docker container (UBI 9.6 for RHEL 9.6 compatibility)
- Download compatible wheels and source distributions
- Create zip file in `build-output/`

## Output

The built package is placed in the `build-output/` directory:
- `chunksilo-{VERSION}-manylinux_2_34_x86_64.zip`

## Cleanup

To clean up build artifacts:

```bash
rm -rf release_common/ release_package_manylinux_2_34/ build-output/
```

## Notes

- The scripts use the same Docker images as the GitHub Actions workflow
- The packaging process includes multiple fallback strategies for downloading dependencies
- Source distributions are allowed as a fallback if wheels aren't available
- The scripts preserve all the logic from the original workflow, including the legacy pip resolver usage