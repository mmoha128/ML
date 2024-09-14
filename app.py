from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('house_model.pkl', 'rb'))

# Load and train the model
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# Split the dataset
X = df.drop('PRICE', axis=1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Define a route for home
@app.route('/')
def home():
    return "Welcome to the House Price Prediction API!"

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the user
    input_data = request.json
    input_features = np.array([input_data['MedInc'], input_data['HouseAge'], input_data['AveRooms'],
                               input_data['AveBedrms'], input_data['Population'], input_data['AveOccup'],
                               input_data['Latitude'], input_data['Longitude']]).reshape(1, -1)
    
    # Make a prediction using the model
    prediction = model.predict(input_features)

    # Return the prediction as a JSON response
    return jsonify({'predicted_price': prediction[0]})



if __name__ == '__main__':
    app.run(debug=True)

