import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import json

# Set page config
st.set_page_config(page_title="Car Price Prediction", layout="wide", initial_sidebar_state="expanded", page_icon="🚗")

# Load the model, columns, and feature data
@st.cache_resource
def load_model_and_columns():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('input_columns.pkl', 'rb') as file:
        columns = pickle.load(file)
    return model, columns

@st.cache_data
def load_feature_data():
    # Load pre-processed feature data from a JSON file
    with open('feature_data.json', 'r') as file:
        feature_data = json.load(file)
    return feature_data

# Load the model, columns, and feature data
model, columns = load_model_and_columns()
feature_data = load_feature_data()

# Load the image
image = Image.open("feixiao.png")

# Define the features with descriptions
features = {
    'year': 'Year of manufacture',
    'make': 'Manufacturer',
    'model': "Car's model",
    'trim': 'Levels of features and equipment',
    'body': 'Main structure of the vehicle that sits on the frame',
    'transmission': 'Gearbox',
    'state': 'Where the car is registered',
    'condition': 'Vehicle condition (1-49) higher is better',
    'odometer': 'Total distance traveled',
    'color': 'Exterior paintwork of the car',
    'interior': 'Design and materials used inside the car cabin',
    'seller': 'Who sold the vehicle',
    'season': 'Season of the year'
}

# Create the Streamlit app
st.title('Car Price Prediction App')

# Create input fields for each feature
user_input = {}

for feature, description in features.items():
    if feature == 'year':
        user_input[feature] = st.number_input(f"{feature.capitalize()} - {description}", min_value=1900, max_value=2024, step=1, value=2024)
    elif feature == 'make':
        user_input[feature] = st.selectbox(f"{feature.capitalize()} - {description}", feature_data[feature])
    elif feature == 'model':
        models = feature_data['model_by_make'].get(user_input['make'], [])
        user_input[feature] = st.selectbox(f"{feature.capitalize()} - {description}", models)
    elif feature == 'trim':
        trims = feature_data['trim_by_make_model'].get(f"{user_input['make']}_{user_input['model']}", [])
        user_input[feature] = st.selectbox(f"{feature.capitalize()} - {description}", trims)
    elif feature == 'odometer':
        user_input[feature] = st.number_input(f"{feature.capitalize()} - {description}", min_value=0, max_value=1000000, step=1000)
    elif feature == 'condition':
        user_input[feature] = st.slider(f"{feature.capitalize()} - {description}", min_value=1, max_value=49, value=25)
    else:
        user_input[feature] = st.selectbox(f"{feature.capitalize()} - {description}", feature_data[feature])

# Create a button to make predictions
if st.button('Predict Car Price'):
    # Prepare the input data
    input_data = pd.DataFrame([user_input])
    
    # Ensure the input data has all the columns the model expects
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder the columns to match the model's expected input
    input_data = input_data[columns]
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.success(f'The predicted price of the car is ${prediction[0]:,.2f}')
    
    # Display the image
    st.image(image, caption='Hooray! Prediction complete.', use_column_width=True)
    
    # Display error information
    mse = 25919577.497542154  
    rmse = np.sqrt(mse)
    st.info(f'Note: The prediction has a root mean square error of ${rmse:,.2f}. '
            f'This means the actual price could be roughly ${rmse:,.2f} higher or lower than the prediction.')

# Add some information about the app
st.info("This project is made for Epsilon AI's final project.")