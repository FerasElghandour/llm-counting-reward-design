# Reproduction and benchmark scope

The complete original code-cell workflow is restored with source provenance and an explicit CUDA preflight. Its five-step then80-step phases continue the same adapter; they are not independent experiments.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

The current host has no CUDA GPU. The recovered Unsloth/vLLM/TRL workflow has not been rerun or dependency-validated here, so no new trained-LLM result is claimed.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Recorded command definitions

### checks

```text
{python} smoke_test.py
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
