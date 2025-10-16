# Clone the repo
git clone https://github.com/bnsairam/predict_aqi.py.git
cd predict_aqi.py

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # (On Windows)
# source venv/bin/activate   # (On Mac/Linux)

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn joblib

# Run the AQI predictor script
python predict_aqi.py

# After training, your model will be saved as:
# aqi_predictor_model.pkl
