# Contributing

Thanks for taking a look. This is a research prototype, so contributions are welcome as long as they keep the repo **auditable** and **honest**.

## Ground rules
- No fabricated claims or results. If you add numbers, add the exact command / script used to produce them.
- Keep changes small and well-scoped.
- Prefer clarity over cleverness.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

Fast tests (offline):

```bash
pytest -m "not slow"
```

Slow HuggingFace integration (downloads a model):

```bash
RUN_SLOW=1 pytest
```

## Coding style
- If you change the math, add a short explanation and a unit test.
- Keep docstrings practical: describe inputs/outputs and the key assumptions.

## Submitting a PR
- Describe what changed and why.
- Mention any behavior changes (even if minor).
- Include updated docs/tests if relevant.

