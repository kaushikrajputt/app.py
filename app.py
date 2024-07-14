from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_file = 'linear_reg_model.pkl'
model = joblib.load(model_file)

# Initialize LabelEncoders for categorical variables
label_encoders = {}
categorical_cols = ['education', 'city', 'gender', 'everbenched', 'department']

# Example data for fitting encoders (you should use your actual data)
data = pd.read_csv('employeee.csv')

# Fit and transform categorical columns in the training data
for column in categorical_cols:
    label_encoders[column] = LabelEncoder()
    label_encoders[column].fit(data[column])  # Fit with your actual data here

# Initialize Flask application
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()

    # Prepare data for prediction
    new_data = pd.DataFrame({
        'education': [data['education']],
        'joiningyear': [int(data['joiningyear'])],
        'city': [data['city']],
        'paymenttier': [int(data['paymenttier'])],
        'age': [int(data['age'])],
        'gender': [data['gender']],
        'everbenched': [data['everbenched']],
        'experienceincurrentdomain': [int(data['experienceincurrentdomain'])],
        'department': [data['department']]
    })

    # Transform categorical variables using the fitted encoders
    for column in categorical_cols:
        new_data[column] = label_encoders[column].transform(new_data[column])

    # Predict using the model
    prediction = model.predict(new_data)

    # Return prediction as JSON response
    return jsonify({'prediction': prediction[0]})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
