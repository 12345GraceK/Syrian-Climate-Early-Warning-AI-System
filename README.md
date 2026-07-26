# 🛰️ Syrian Climate Early-Warning AI System (SDG 13 & SDG 2)
An automated data engineering pipeline and predictive AI early-warning system built to monitor, analyze, and forecast localized agricultural drought anomalies in the Al-Hasakah sector of Northeast Syria. This project explicitly bridges advanced machine learning frameworks with international development mandates.
## 📌 Strategic UN Alignment
This system is designed as a digital innovation asset to support proactive humanitarian resource allocation before environmental crises escalate:
* **SDG 13: Climate Action** – Implementing predictive early-warning infrastructures to mitigate the risks of extreme, climate-induced weather anomalies.
* **SDG 2: Zero Hunger** – Securing agricultural resilience by identifying stress parameters in core winter wheat and barley crops to prevent regional food insecurity.
## 🏗️ System Architecture
The application runs as a continuous data processing pipeline split into five distinct functional layers:
* **Data Ingestion** – Streams raw JSON weather and soil attributes directly from the Open-Meteo API.
* **ETL & Resampling Engine** – Aggregates 24-hour fractional data arrays into clean, daily-mean database matrices using Pandas.
* **Feature Engineering Pipeline** – Computes localized 14-day rolling rain deficits to evaluate intensifying agricultural stress.
* **Predictive AI Engine** – Utilizes a 100-tree Random Forest Classifier trained on a 7-day look-ahead horizon to predict anomalies without data leakage.
* **Operational Mission Dashboard** – Feeds machine learning probability outputs into a Streamlit web portal for easy field use.
1. **Historical ETL Pipeline (`pipeline.py`):** Ingests 731 continuous daily observations of archive historical telemetry data from the Open-Meteo REST API, mapping coordinates to Syrian crop hubs.
2. **Feature Engineering Engine:** Programmatically handles null inputs using forward-fills, computes a **14-day rolling precipitation deficit**, and defines a compound agricultural stress target label.
3. **Model Optimization (`train_and_save.py`):** Trains a 100-estimator Random Forest Classifier. Implements a **7-day chronological look-ahead shift** to break immediate mathematical leakage, training the AI to recognize patterns ahead of time.
4. **Live Forecasting Pipeline (`live_pipeline.py`):** Pulls active, real-time forecast data. To bypass resolution mismatches (as soil moisture data is only available hourly), it implements **Pandas time-series resampling** to downsample 24-hour records into clean daily averages.
5. **Operational UI Dashboard (`app.py`):** Translates complex machine learning array probabilities into a 0-100% predictive risk visualization map optimized for non-technical field workers.
## 🧠 AI Model Analytics & Evaluation
By shifting our target labels by 7 days, the model successfully avoids **Data Leakage**, transforming from a static rule-checker into a genuine predictive engine.
### Performance Report (7-Day Predictive Horizon)
* **Overall Accuracy:** 91%
* **Drought Class Precision (90%):** Ensures that when the system triggers a drought warning, it is correct 90% of the time, preventing the waste of scarce UN field resources.
* **Drought Class Recall (84%):** Guarantees the AI successfully catches 84% of upcoming crop failures a week in advance, prioritizing the elimination of catastrophic false negatives.
## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.11+
* **Frameworks & Data Libraries:** Pandas, NumPy, Scikit-Learn
* **Serialization:** Pickle
* **User Interface & Delivery:** Streamlit
* **Infrastructure & Sources:** Open-Meteo Climatology APIs
## 🚀 Installation & Local Execution
### 1. Clone & Environment Setup
Open your terminal (or Anaconda Prompt) and navigate to the folder:
```
cd C:\Users\grace\climate-undp-tracker
pip install requests pandas scikit-learn streamlit
```
### 2. Run the Historical Ingestion Pipeline
Download the historical archive data blocks to generate your local dataset:
```
python pipeline.ipynb
```
### 3. Train and Validate the AI Model
Train the Random Forest architecture and serialize the fitted blueprint file:
```
python model.ipynb
```
### 4. Fetch the Active 2026 Live Forecast
Pull the current week's telemetry streams and run the hourly-to-daily time resampling engine:
```
python live_pipeline.ipynb
```
### 5. Launch the Humanitarian Mission Control Dashboard
Deploy the interactive Streamlit user environment locally:
```
streamlit run app.py
```
