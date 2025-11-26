# Weather Condition Predictor (India)

This project demonstrates a simple artificial neural network (ANN) that predicts the dominant short-term weather condition for regions in India. A Flask web application exposes the trained model for interactive use and via a JSON API. A synthetic—but climatologically inspired—dataset is generated on the fly so the project can be run end-to-end without external data.

## Project structure

```
.
├── app.py                 # Flask app exposing web UI + REST API
├── train.py               # Script to generate data, train, and persist the ANN
├── requirements.txt       # Python dependencies
├── data/                  # Synthetic dataset (created after training)
├── models/                # Saved model and preprocessing artifacts
├── static/css/            # Styles for the HTML frontend
└── templates/             # HTML templates for the Flask app
```

## Quick start

1. **Create a virtual environment (recommended)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Generate data and train the ANN** (produces `models/weather_ann.keras`, `models/feature_transformer.joblib`, and `models/label_encoder.joblib`)
   ```powershell
   python train.py
   ```

4. **Start the Flask development server**
   ```powershell
   python app.py
   ```

5. **Open the app** at [http://127.0.0.1:5000](http://127.0.0.1:5000) to submit predictions, or send a POST request to `http://127.0.0.1:5000/api/predict` with JSON payloads.

## Example API request

```powershell
curl -X POST http://127.0.0.1:5000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{
        \"region\": \"South\",
        \"month\": 7,
        \"temperature_c\": 29.5,
        \"humidity_pct\": 82,
        \"pressure_hpa\": 1002,
        \"wind_speed_kph\": 14,
        \"precip_mm\": 18,
        \"cloud_cover_pct\": 78
      }"
```

Response:

```json
{
  "condition": "Rain",
  "confidence": 0.73,
  "probabilities": {
    "Cloudy": 0.18,
    "Fog": 0.02,
    "Rain": 0.73,
    "Storm": 0.05,
    "Sunny": 0.02
  }
}
```

## Notes

- The model, scaler, and label encoder are stored under `models/` so the Flask app can reuse them without retraining.
- Synthetic data is generated deterministically (via a fixed random seed) for reproducibility while still reflecting regional and seasonal variability across India.
- Retrain whenever you adjust the synthetic data generation heuristics or the network architecture.

## Testing and linting

The project is intentionally lightweight. No automated tests are provided, but you can quickly sanity-check predictions using the API example above once the server is running.
