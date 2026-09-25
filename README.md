# Mycelium-Based AI Integration

*Giving machine learning a body through fungi.*

A research concept for using living mycelium, the root-like network of fungi, as a physical, self-repairing substrate for AI, plus a small scikit-learn script that predicts mycelium growth from environmental readings.

> **Status: concept paper and early Python prototype.** No mycelium has been wired to a computer in this repository. What runs today is a growth-prediction regression script and a FastAPI endpoint that serves its model. You supply the data: no dataset is included. Most files in the `src/`, `docker/`, `notebooks/` and `sql/` folders are empty placeholders.

<img src="docs/Viz%20Graph.png" alt="Layered architecture diagram" width="700"/>

## Why mycelium?

Mycelium conducts electrical signals, grows and reroutes around damage, and reacts to its environment. There is real research on fungal electrical spiking and on fungi as unconventional computing substrates. This project asks what an AI system would look like if part of it lived in such a network: decentralized, low-energy and able to repair itself.

The proposed system has four layers (see the diagram above):

1. **Biological layer:** mycelium interfaced with electrodes and environmental stimuli.
2. **Interface layer:** microcontrollers that read signals and apply stimuli.
3. **Computation layer:** models that interpret signals, classify states and generate feedback.
4. **Output layer:** APIs and HTML dashboards.

## What the code does

| File | What it does |
| --- | --- |
| `ML growth patterns.py` | Loads a CSV, trains Linear, Ridge, Lasso and Random Forest regressors to predict `growth` from `time, temperature, humidity, ph, light_intensity, co2_level, substrate_type`, reports 5-fold CV R^2, saves the best model to `mycelium_growth_model.pkl`, plots predictions, and appends metrics to `model_performance_log.csv` |
| `model_api.py` | FastAPI app that loads `mycelium_growth_model.pkl` and serves `POST /predict/` (the seven features as JSON) and `GET /` |
| `ML Growth Pattern.html` | Static page that fetches `model_performance_log.csv` and renders it as a table |
| `Ml.sql` | Despite the extension, a one-row CSV in the log format from an earlier run. The dataset it came from is not in the repository, so the numbers cannot be reproduced |
| `Self-Repairing AI Systems` | Draft concept paper: *A Mycelium-Based Framework for Decentralized, Self-Repairing AI Systems* (plain text) |
| `ANGELCORE_x_Mycelium_No_Links.pdf` | Concept note tying this project into ANGELCORE, a speculative framework with RAVEN (reasoning), SERAPH (ethics) and THRONOS (physical) layers |
| `docs/Viz Graph.png` | Architecture diagram |

Empty placeholder files (0 or 1 byte): everything under `src/`, `docker/`, `notebooks/`, `sql/`, `experiments/`, and `environment.yml`.

## Quick start

Requires Python 3.9+. You need your own CSV with the columns listed above; all values must be numeric, including `substrate_type` (encode it as a number).

```bash
git clone https://github.com/Mattbusel/Mycelium-Based-AI-Integration.git
cd Mycelium-Based-AI-Integration

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib joblib fastapi uvicorn
```

The script reads from a hard-coded path, `DATA_PATH = 'path_to_dataset.csv'`. Either name your file `path_to_dataset.csv` or edit that line, then:

```bash
python "ML growth patterns.py"      # trains, shows a plot, writes mycelium_growth_model.pkl
uvicorn model_api:app --reload      # serves the model on http://127.0.0.1:8000
```

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"time": 7, "temperature": 24, "humidity": 85, "ph": 6.0, "light_intensity": 0, "co2_level": 800, "substrate_type": 1}'
```

To view the metrics table, serve the folder (for example `python -m http.server`) and open `ML Growth Pattern.html`; opening it straight from disk blocks the CSV fetch in most browsers.

## Roadmap

| Phase | Goal |
| --- | --- |
| 1 | Grow mycelium samples and measure signal conduction; design basic interfaces |
| 2 | Connect samples to sensors and microcontrollers (not started in code) |
| 3 | Train models on real mycelium signals and close a feedback loop |
| 4 | Explore robotics, environmental sensing and adaptive materials |

Phase 1 cultivation work was described as under way in earlier notes; no data from it is in the repository yet.

## Limitations

- No bio-interface code. `sensor_control.py` and `mycelium_signals.py` are empty.
- No dataset, so the one logged result (`Ml.sql`) cannot be checked.
- No `requirements.txt`; the package list above is taken from the imports.
- `influence_growth` in the training script only multiplies a prediction by a factor; it does not act on anything physical.

## Contributing

Mycologists, electrophysiology people and ML engineers are all welcome. The most valuable contribution would be a small, real dataset of mycelium electrical recordings or growth measurements.
