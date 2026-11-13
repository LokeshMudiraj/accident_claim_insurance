# Accident Claim Insurance - FastAPI & Analytics Project

This project predicts accident insurance claim approvals using Random Forest and provides an interactive FastAPI dashboard along with visualization tools.

## Structure
- `app.py`: FastAPI server & prediction endpoints.
- `models/model.py`: Model training & pickle serialization script.
- `eda/exploratory_analysis.py`: Seaborn & Matplotlib EDA script.
- `templates/dashboard.html`: Interactive web interface with Plotly charts.
- `data/accident_claims.csv`: Sample dataset.

## How to Run

1. **Activate venv and Install Dependencies**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Model**:
   ```bash
   python models/model.py
   ```

3. **Run Application**:
   ```bash
   uvicorn app:app --reload
   ```

4. **Access Web App**:
   Open browser at `http://127.0.0.1:8000/`
