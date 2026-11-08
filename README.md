# Mycelium-Based AI Integration

*Giving machine learning a body through fungi.*

An idea: use living mycelium (the root-like network of fungi) as a physical, self-repairing base for AI. Plus a small scikit-learn script that predicts mycelium growth from environment readings.

> **Status: concept paper and early Python prototype.** No mycelium has been wired to a computer here. What runs today is a growth-prediction script and a FastAPI endpoint for its model. You bring the data. No dataset is included. Most files in `src/`, `docker/`, `notebooks/` and `sql/` are empty placeholders.

<img src="docs/Viz%20Graph.png" alt="Layered architecture diagram" width="700"/>

## Why mycelium?

Mycelium carries electrical signals, grows around damage, and reacts to its surroundings. There is real research on fungal electrical spiking and on fungi as computers. This project asks what AI would look like if part of it lived in that network: spread out, low-energy, able to repair itself.

The planned system has four layers (diagram above):

1. **Biological layer:** mycelium hooked to electrodes and stimuli.
2. **Interface layer:** microcontrollers that read signals and apply stimuli.
3. **Computation layer:** models that read signals, label states and send feedback.
4. **Output layer:** APIs and HTML dashboards.

## What the code does

| File | What it does |
| --- | --- |
| `ML growth patterns.py` | Loads a CSV, trains Linear, Ridge, Lasso and Random Forest regressors to predict `growth` from `time, temperature, humidity, ph, light_intensity, co2_level, substrate_type`, reports 5-fold CV R^2, saves the best model to `mycelium_growth_model.pkl`, plots predictions, and appends metrics to `model_performance_log.csv` |
| `model_api.py` | FastAPI app that loads `mycelium_growth_model.pkl` and serves `POST /predict/` (the seven features as JSON) and `GET /` |
| `ML Growth Pattern.html` | Static page that fetches `model_performance_log.csv` and renders it as a table |
| `Ml.sql` | Not SQL. A one-row CSV in the log format from an earlier run. Its dataset is not in the repo, so the numbers can't be rerun |
| `Self-Repairing AI Systems` | Draft concept paper: *A Mycelium-Based Framework for Decentralized, Self-Repairing AI Systems* (plain text) |
| `ANGELCORE_x_Mycelium_No_Links.pdf` | Concept note tying this project into ANGELCORE, a speculative framework with RAVEN (reasoning), SERAPH (ethics) and THRONOS (physical) layers |
| `docs/Viz Graph.png` | Architecture diagram |

Empty placeholder files (0 or 1 byte): everything under `src/`, `docker/`, `notebooks/`, `sql/`, `experiments/`, and `environment.yml`.

## Quick start

Needs Python 3.9+ and your own CSV with the columns above. All values must be numbers, including `substrate_type`.

```bash
git clone https://github.com/Mattbusel/Mycelium-Based-AI-Integration.git
cd Mycelium-Based-AI-Integration

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib joblib fastapi uvicorn
```

The script reads a hard-coded path, `DATA_PATH = 'path_to_dataset.csv'`. Name your file `path_to_dataset.csv` or edit that line, then:

```bash
python "ML growth patterns.py"      # trains, shows a plot, writes mycelium_growth_model.pkl
uvicorn model_api:app --reload      # serves the model on http://127.0.0.1:8000
```

```bash
curl -X POST http://127.0.0.1:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{"time": 7, "temperature": 24, "humidity": 85, "ph": 6.0, "light_intensity": 0, "co2_level": 800, "substrate_type": 1}'
```

To see the metrics table, serve the folder (for example `python -m http.server`) and open `ML Growth Pattern.html`. Most browsers block the CSV fetch if you open it straight from disk.

## Roadmap

| Phase | Goal |
| --- | --- |
| 1 | Grow mycelium samples and measure signal conduction; design basic interfaces |
| 2 | Connect samples to sensors and microcontrollers (not started in code) |
| 3 | Train models on real mycelium signals and close a feedback loop |
| 4 | Explore robotics, environmental sensing and adaptive materials |

Earlier notes said Phase 1 growing had started. No data from it is in the repo yet.

## Limitations

- No bio-interface code. `sensor_control.py` and `mycelium_signals.py` are empty.
- No dataset, so the one logged result (`Ml.sql`) cannot be checked.
- No `requirements.txt`. The package list above comes from the imports.
- `influence_growth` in the training script just multiplies a prediction by a factor. It touches nothing physical.

## Contributing

Mycologists, electrophysiology people and ML engineers welcome. The most useful thing would be a small, real dataset of mycelium electrical recordings or growth measurements.
