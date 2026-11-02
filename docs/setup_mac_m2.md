# Setup on Mac M2 (Apple Silicon)

```bash
# 1. Python 3.11 or 3.12 (Homebrew)
brew install python@3.12 git
# 2. Get the code
git clone https://github.com/liliarouhi/qaoa-warmstart.git
cd qaoa-warmstart
# 3. Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 4. Install (core + PennyLane check + test tools)
pip install -e ".[quantum,dev]"
# 5. Run the tests and the quickstart
pytest
python scripts/quickstart.py
# 6. Later (Phase 3): machine-learning extras; PyTorch has native arm64 wheels
pip install -e ".[ml]"
```

Expected: all tests pass (51 at v0.2.0). If `pytest` reports a skipped PennyLane test,
step 4 did not install the `quantum` extra.
SQLite R*Tree: included in Homebrew and python.org builds; `tests/test_db.py` checks it.
