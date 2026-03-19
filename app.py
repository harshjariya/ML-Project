from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved pipeline
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Collect data from Form
        input_data = {
            'age': int(request.form['age']),
            'job': request.form['job'],
            'marital': request.form['marital'],
            'balance': int(request.form['balance']),
            'housing': request.form['housing'],
            'loan': request.form['loan'],
            'duration': int(request.form['duration']),
            
            # 2. Add default values for columns the model expects but aren't in the form
            'education': 'secondary',
            'default': 'no',
            'contact': 'cellular',
            'day': 15,
            'month': 'may',
            'campaign': 1,
            'pdays': -1,
            'previous': 0,
            'poutcome': 'unknown'
        }
        
        # 3. Convert to DataFrame
        query_df = pd.DataFrame([input_data])
        
        # 4. Predict
        prediction = model.predict(query_df)[0]
        
        # 5. Result Logic
        if prediction == 1:
            result = "Success! The client is likely to subscribe. ✅"
        else:
            result = "The client is unlikely to subscribe. ❌"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        # This will show you the exact error on the webpage if it fails again
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)