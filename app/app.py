import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the scaler and model
try:
    sc = pickle.load(open('sc.pkl', 'rb'))
    model = pickle.load(open('classifier.pkl', 'rb'))
except FileNotFoundError:
    print("Error: Model or scaler file not found. Make sure the files exist in the correct paths.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form and convert to float
    float_features = [float(x) for x in request.form.values()]
    
    # Transform input features using the loaded scaler
    final_features = [np.array(float_features)]
    transformed_features = sc.transform(final_features)
    
    # Make predictions using the loaded model
    pred = model.predict(transformed_features)
    
    # Pass the prediction to the result template
    return render_template('result.html', prediction=pred)

if __name__ == "__main__":
    # Run the Flask application in debug mode
    app.run(debug=True)



