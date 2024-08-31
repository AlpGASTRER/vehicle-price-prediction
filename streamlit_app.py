import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the data, model, and column names
@st.cache_data
def load_data():
    df = pd.read_csv('final_csv.csv')
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('columns.pkl', 'rb') as file:
        columns = pickle.load(file)
    return df, model, columns

df, model, columns = load_data()

# Define the features
features = ['year', 'make', 'model', 'trim', 'body', 'transmission', 'state', 'condition', 'odometer', 'color', 'interior', 'seller', 'season']

# Create the Streamlit app
st.title('Car Price Prediction App')

# Create input fields for each feature
user_input = {}

# Year
user_input['year'] = st.selectbox('Year', sorted(df['year'].unique(), reverse=True))

# Make
user_input['make'] = st.selectbox('Make', sorted(df['make'].unique()))

# Model (filtered by make)
models = df[df['make'] == user_input['make']]['model'].unique()
user_input['model'] = st.selectbox('Model', sorted(models))

# Trim (filtered by make and model)
trims = df[(df['make'] == user_input['make']) & (df['model'] == user_input['model'])]['trim'].unique()
user_input['trim'] = st.selectbox('Trim', sorted(trims))

# Body
user_input['body'] = st.selectbox('Body', sorted(df['body'].unique()))

# Transmission
user_input['transmission'] = st.selectbox('Transmission', sorted(df['transmission'].unique()))

# State
user_input['state'] = st.selectbox('State', sorted(df['state'].unique()))

# Condition
user_input['condition'] = st.selectbox('Condition', sorted(df['condition'].unique()))

# Odometer (continuous value)
user_input['odometer'] = st.number_input('Odometer', min_value=0, max_value=1000000, step=1000)

# Color
user_input['color'] = st.selectbox('Color', sorted(df['color'].unique()))

# Interior
user_input['interior'] = st.selectbox('Interior', sorted(df['interior'].unique()))

# Seller
user_input['seller'] = st.selectbox('Seller', sorted(df['seller'].unique()))

# Season
user_input['season'] = st.selectbox('Season', sorted(df['season'].unique()))

# Create a button to make predictions
if st.button('Predict Price'):
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

# Add some information about the app
st.info('This app predicts the price of a car based on various features. Select the options above and click "Predict Price" to get an estimate.')
