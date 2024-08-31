import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import json
from sklearn.preprocessing import LabelEncoder

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

# Define the features with descriptions and types
features = {
    'year': {'description': 'Year of manufacture', 'type': 'numeric'},
    'make': {'description': 'Manufacturer', 'type': 'categorical'},
    'model': {'description': "Car's model", 'type': 'categorical'},
    'trim': {'description': 'Levels of features and equipment', 'type': 'categorical'},
    'body': {'description': 'Main structure of the vehicle that sits on the frame', 'type': 'categorical'},
    'transmission': {'description': 'Gearbox', 'type': 'categorical'},
    'state': {'description': 'Where the car is registered', 'type': 'categorical'},
    'condition': {'description': 'Vehicle condition (1-49) the higher the better', 'type': 'numeric'},
    'odometer': {'description': 'Total distance traveled', 'type': 'numeric'},
    'color': {'description': 'Exterior paintwork of the car', 'type': 'categorical'},
    'interior': {'description': 'Design and materials used inside the car cabin', 'type': 'categorical'},
    'seller': {'description': 'Who sold the vehicle', 'type': 'categorical'},
    'season': {'description': 'Season of the year', 'type': 'categorical'}
}
# Create the Streamlit app
st.title('Car Price Prediction App')

# Verify columns match features
column_set = set(columns)
feature_set = set(features.keys())

if column_set != feature_set:
    st.error("Mismatch between columns and features!")
    st.write("Columns not in features:", column_set - feature_set)
    st.write("Features not in columns:", feature_set - column_set)
    st.stop()

# Create input fields for each feature
user_input = {}

for feature, info in features.items():
    if info['type'] == 'numeric':
        if feature == 'year':
            user_input[feature] = st.number_input(f"{feature.capitalize()} - {info['description']}", min_value=1900, max_value=2024, value=2024)
        elif feature == 'condition':
            user_input[feature] = st.slider(f"{feature.capitalize()} - {info['description']}", min_value=1, max_value=49, value=25)
        else:
            user_input[feature] = st.number_input(f"{feature.capitalize()} - {info['description']}", min_value=0)
    else:  # categorical
        if feature == 'make':
            user_input[feature] = st.selectbox(f"{feature.capitalize()} - {info['description']}", feature_data[feature])
        elif feature == 'model':
            models = feature_data['model_by_make'].get(user_input['make'], [])
            user_input[feature] = st.selectbox(f"{feature.capitalize()} - {info['description']}", models)
        elif feature == 'trim':
            trims = feature_data['trim_by_make_model'].get(f"{user_input['make']}_{user_input['model']}", [])
            user_input[feature] = st.selectbox(f"{feature.capitalize()} - {info['description']}", trims)
        else:
            user_input[feature] = st.selectbox(f"{feature.capitalize()} - {info['description']}", feature_data[feature])


# Create a button to make predictions
if st.button('Predict Car Price'):
    try:
        # Prepare the input data
        input_data = pd.DataFrame([user_input])
        
        # Encode categorical variables
        le = LabelEncoder()
        for column in input_data.columns:
            if features[column]['type'] == 'categorical':
                input_data[column] = le.fit_transform(input_data[column])
        
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
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.error("Please check your input values and try again.")
        st.error("If the problem persists, there might be an issue with the model or input data format.")

# Add some information about the app
st.info("This project is made for Epsilon AI's final project.")
