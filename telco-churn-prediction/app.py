from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model, scaler, and columns
model = joblib.load('model_rf.pkl')  # Updated for Random Forest
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl') if os.path.exists('model_columns.pkl') else joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    probability = None
    error = None

    if request.method == 'POST':
        try:
            tenure = float(request.form['tenure'])
            monthly_charges = float(request.form['monthly_charges'])
            total_charges = float(request.form['total_charges'])
            contract = int(request.form['contract'])
            internet_service = int(request.form['internet_service'])
            tech_support = int(request.form['tech_support'])

            input_dict = dict.fromkeys(model_columns, 0)
            input_dict['tenure'] = tenure
            input_dict['MonthlyCharges'] = monthly_charges
            input_dict['TotalCharges'] = total_charges

            contract_map = {0: 'Contract_One year', 1: 'Contract_Two year'}
            if contract in contract_map and contract_map[contract] in input_dict:
                input_dict[contract_map[contract]] = 1

            internet_map = {1: 'InternetService_Fiber optic', 0: 'InternetService_DSL'}
            if internet_service in internet_map and internet_map[internet_service] in input_dict:
                input_dict[internet_map[internet_service]] = 1

            if tech_support == 1 and 'TechSupport_Yes' in input_dict:
                input_dict['TechSupport_Yes'] = 1

            input_array = np.array([list(input_dict.values())])
            scaled_input = scaler.transform(input_array)

            pred = model.predict(scaled_input)[0]
            prediction = int(pred)
            probability = round(model.predict_proba(scaled_input)[0][1] * 100, 2)

        except Exception as e:
            error = f"❌ Error: {e}"

    return render_template('predict.html', prediction=prediction, probability=probability, error=error)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)


