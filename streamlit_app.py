import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# Set page config for dark mode
st.set_page_config(page_title="Car Price Prediction", layout="wide", initial_sidebar_state="expanded", page_icon="🚗")

# Apply dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #2b2b2b;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Load the data, model, and column names
@st.cache_data
def load_data():
    df = pd.read_csv('your_data.csv')
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('columns.pkl', 'rb') as file:
        columns = pickle.load(file)
    return df, model, columns

df, model, columns = load_data()

# Load the image
image = Image.open("feixaio.png")

# Define the features with descriptions
features = {
    'year': 'Year of manufacture',
    'make': 'Manufacturer',
    'model': "Car's model",
    'trim': 'Levels of features and equipment',
    'body': 'Main structure of the vehicle that sits on the frame',
    'transmission': 'Gearbox',
    'state': 'Where the car is registered',
    'condition': 'Vehicle condition (1-49)',
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
    st.subheader(f"{feature.capitalize()} - {description}")
    
    if feature == 'year':
        user_input[feature] = st.number_input(f"Enter {description}", min_value=1900, max_value=2024, step=1, value=2024)
    elif feature == 'make':
        user_input[feature] = st.selectbox(f"Select {description}", sorted(df[feature].unique()))
    elif feature == 'model':
        models = df[df['make'] == user_input['make']][feature].unique()
        user_input[feature] = st.selectbox(f"Select {description}", sorted(models))
    elif feature == 'trim':
        trims = df[(df['make'] == user_input['make']) & (df['model'] == user_input['model'])][feature].unique()
        user_input[feature] = st.selectbox(f"Select {description}", sorted(trims))
    elif feature == 'odometer':
        user_input[feature] = st.number_input(f"Enter {description}", min_value=0, max_value=1000000, step=1000)
    elif feature == 'condition':
        user_input[feature] = st.slider(f"Select {description}", min_value=1, max_value=49, value=25)
    else:
        user_input[feature] = st.selectbox(f"Select {description}", sorted(df[feature].unique()))
    
    st.write(f"Disclaimer: This {feature} information is used to estimate the car's price and may affect the prediction.")

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
st.info('This app predicts the price of a car based on various features. Fill in the details above and click "Predict Car Price" to get an estimate.')