# Mycelium-Based AI Integration

Concept: living mycelium as a self-repairing base for AI.

Concept paper and early Python prototype. No mycelium is wired up yet.

- Predicts mycelium growth from environment readings
- Compares 4 scikit-learn models
- Serves the best one with FastAPI
- Includes the draft concept paper

## Install

```bash
pip install pandas numpy scikit-learn matplotlib joblib fastapi uvicorn
```

## Use

```bash
python "ML growth patterns.py"
uvicorn model_api:app --reload
```

Bring your own CSV, saved as `path_to_dataset.csv`.

More: [docs/DETAILS.md](docs/DETAILS.md)

License: MIT
