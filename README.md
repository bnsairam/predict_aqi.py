# Clone the repository
git clone https://github.com/bnsairam/predict_aqi.py.git
cd predict_aqi.py

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# Run the script
python predict_aqi.py

# ✅ Model will be saved as:
# aqi_predictor_model.pkl
