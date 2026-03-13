# Backend

AI-powered precision agriculture backend built with FastAPI.

## Quick Start

```bash
cd agrosense-backend

# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OpenWeatherMap API key

# 3. Train models (generates models/*.pkl using synthetic data)
python training/train_irrigation.py
python training/train_recommend.py
# Optional (requires TensorFlow + PlantVillage dataset):
# python training/train_disease.py

# 4. Run the server
uvicorn main:app --reload --port 8000
```

The server will be available at: **http://127.0.0.1:8000**

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/irrigation/predict` | Predict irrigation requirement |
| POST | `/api/health/analyze` | Analyze crop disease from image |
| POST | `/api/health/analyze-batch` | Batch analysis (up to 10 images) |
| POST | `/api/recommend/crop` | Recommend crops for a region |
| GET | `/api/weather?location=Nagpur` | Fetch weather data |
| GET | `/api/calamity/recent?state=Maharashtra` | Fetch recent calamity data |

---

## ML Models

| Model | File | Dataset | Algorithm |
|-------|------|---------|-----------|
| Irrigation | `models/irrigation_model.pkl` | Kaggle — Plant Watering | Random Forest Regressor |
| Disease Detection | `models/disease_model.h5` | PlantVillage (Kaggle) | MobileNetV2 (Transfer Learning) |
| Crop Recommendation | `models/recommend_model.pkl` | Crop Recommendation (Kaggle) | Random Forest Classifier |

> Training scripts in `training/` use **synthetic data** by default (same schema as Kaggle datasets).
> Replace with real downloaded CSVs for production accuracy.

---

## Project Structure

```
agrosense-backend/
├── main.py                     # FastAPI app, middleware, model loader
├── requirements.txt
├── .env.example
├── routes/
│   ├── irrigation.py           # POST /api/irrigation/predict
│   ├── crop_health.py          # POST /api/health/analyze[,-batch]
│   └── crop_recommend.py       # POST /api/recommend/crop
├── schemas/
│   ├── irrigation.py           # Pydantic request/response models
│   ├── crop_health.py
│   └── crop_recommend.py
├── models/                     # Trained model files (.pkl, .h5)
├── utils/
│   ├── weather.py              # OpenWeatherMap + 30-min cache
│   ├── calamity.py             # Calamity lookup by Indian state
│   └── image_utils.py          # Image preprocessing for CNN
└── training/
    ├── train_irrigation.py
    ├── train_disease.py        # Requires TensorFlow + PlantVillage data
    └── train_recommend.py
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENWEATHER_API_KEY` | Free API key from openweathermap.org |
| `MODEL_DIR` | Path to model directory (default: `./models`) |

---

## Frontend Integration

The AgroSense HTML frontend (`../index.html`) is configured to call this backend at `http://127.0.0.1:8000`.
Make sure the backend is running before using the frontend forms.
