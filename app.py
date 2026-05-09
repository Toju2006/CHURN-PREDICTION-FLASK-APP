from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your churn model
try:
    model = joblib.load('churn_model.joblib')
except Exception as e:
    print(f"Model load error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(x) for x in request.form.values()]
        extented_values = values + [0] * 20
        array = [np.array(extented_values)]
        prediction = model.predict(array)
        
        output = "Likely to Churn" if prediction[0] == 1 else "Not Likely to Churn"
        return render_template('index.html', prediction_text=f'Result: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
