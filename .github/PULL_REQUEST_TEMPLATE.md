## Summary

<!-- What does this PR add or change? -->

## Component type

- [ ] Forecaster (`deepbullwhip/forecast/`)
- [ ] Policy (`deepbullwhip/policy/`)
- [ ] Demand generator (`deepbullwhip/demand/`)
- [ ] Other

## Checklist

- [ ] Component file with `@register()` decorator
- [ ] Updated module `__init__.py`
- [ ] Unit test in `tests/`
- [ ] Optional dependencies added to `pyproject.toml` (if any)
- [ ] If heavy optional deps (torch, gluonts): tests guarded with `pytest.mark.skipif`

## Benchmark output

<!-- Paste output of: python benchmarks/run_leaderboard.py -->

```
<paste benchmark output here>
```

## Test plan

- [ ] `python -m pytest tests/ -v`
- [ ] `python benchmarks/run_leaderboard.py`
