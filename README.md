# Tech Threads – Outfit Match

Tech Threads is a fashion-tech web app that evaluates your outfit based on your mood, event, and local weather, providing a match percentage, trend analysis, and personalized fashion suggestions.

It combines machine learning, trend insights, and user-driven data to help you stay stylish and on-trend.

---

## Features

- **Outfit Match Score**  
  Hybrid score combining ML predictions and trend analysis.

- **Trend Score**  
  Uses Google Trends to evaluate outfit popularity.

- **ML Score**  
  Random Forest model trained on synthetic + real user data.

- **Category Assignment**  
  Classifies outfits as Trendy, Non-Trendy, or Neutral using K-Means clustering.

- **Personalized Suggestions**  
  Recommends adjustments to improve trend alignment.

- **Email Summaries**  
  Sends match percentage and suggestions to users.

- **Power BI Dashboard Ready**  
  Dataset structured for visualization and analytics.

---

## Dataset

- Initially generated **300 synthetic entries** for training.
- Now continuously growing with **real user data**.
- Includes:
  - Weather  
  - Mood  
  - Event  
  - Outfit type  
  - Fabrics  
  - Colors  
  - Match percentages  

Predictions improve over time as more users interact with the app.

---

## ML & Trend Analysis

- **Machine Learning**  
  Random Forest Regressor predicts outfit match percentages.

- **Trend Analysis**  
  Google Trends (PyTrends) for keyword popularity by region.

- **Hybrid Scoring**  
  Combines ML and trend scores into a final match percentage.

- **Fallback Mechanism**  
  Ensures predictions even if trend data is unavailable.

---

## Frontend

Built with **Streamlit** for an interactive UI.

### Features:
- Outfit image selection (`st_img_selectbox`)
- Glass-effect UI with background images
- Metric cards for:
  - Match Score  
  - Trend Score  
  - ML Score  

Users input location, mood, event, and outfit details to receive real-time suggestions.

---

## Future Plans

- **Power BI Dashboard**  
  Visualize trends, top outfits, and user behavior.

- **Dataset Expansion**  
  More users → better predictions.

- **Enhanced Features**  
  - Seasonal recommendations  
  - Fabric-weather insights  
  - Smarter AI suggestions  

---

## Getting Started

### Clone the repo
```bash
git clone <repo-url>
cd <repo-folder>
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Set environment variables (.env)
```
OPENWEATHERAPI=<your_openweather_api_key>
OPENCAGEAPI=<your_opencage_api_key>
EMAIL=<your_email>
PASSWORD=<your_email_password>
```

### Run the app
```bash
streamlit run frontend.py
```

### Add Images
Add outfit images to the working directory to enable selection.

---

## Notes

- Dataset is dynamic (synthetic → real user data).
- ML predictions improve with more data.
- Ensure valid API keys for weather and geocoding services.
- Email requires valid credentials (Gmail may need app access enabled).

---

## Vision

Tech Threads bridges fashion and data science, helping users make smarter, trend-aware outfit choices using AI.
