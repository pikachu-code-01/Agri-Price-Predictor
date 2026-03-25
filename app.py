import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Agri-100 Predictor", page_icon="🌾", layout="wide")
st.title("🌾 AI-ML Mega Price Predictor (100+ Crops)")

# 1. THE 100 CROP LIST
all_crops = [
    "Rice", "Wheat", "Maize", "Barley", "Jowar", "Bajra", "Ragi", "Gram", "Tur", "Moong",
    "Urad", "Lentil", "Peas", "Groundnut", "Rapeseed", "Mustard", "Soybean", "Sunflower", "Sesame", "Cotton",
    "Jute", "Sugarcane", "Tea", "Coffee", "Rubber", "Potato", "Onion", "Tomato", "Cabbage", "Cauliflower",
    "Brinjal", "Okra", "Pumpkin", "Bottle Gourd", "Bitter Gourd", "Cucumber", "Spinach", "Carrot", "Radish", "Garlic",
    "Ginger", "Turmeric", "Coriander", "Cumin", "Black Pepper", "Cardamom", "Clove", "Cinnamon", "Nutmeg", "Apple",
    "Banana", "Mango", "Orange", "Grapes", "Papaya", "Pineapple", "Guava", "Pomegranate", "Watermelon", "Muskmelon",
    "Lemon", "Lime", "Cashew", "Almond", "Walnut", "Pistachio", "Coconut", "Arecanut", "Tobacco", "Opium",
    "Aloe Vera", "Ashwagandha", "Mentha", "Lemongrass", "Rose", "Jasmine", "Marigold", "Tulsi", "Neem", "Bamboo",
    "Mushroom", "Honey", "Silk", "Milk", "Egg", "Wool", "Fish", "Prawn", "Chicken", "Mutton",
    "Saffron", "Vanilla", "Cocoa", "Indigo", "Hemp", "Flax", "Poppy", "Chilli", "Capsicum", "Strawberry"
]

# 2. INPUTS
selected_crop = st.sidebar.selectbox("Choose a Crop", sorted(all_crops))
target_date = st.sidebar.date_input("Select Prediction Date", datetime.now())

# 3. DATA GENERATOR (Simulates 3 years of market data)
@st.cache_data
def generate_data(crop_name):
    np.random.seed(len(crop_name)) 
    dates = pd.date_range(start='2023-01-01', end='2026-12-31', freq='MS')
    base_price = (len(crop_name) * 150) + 800
    prices = base_price + np.random.randint(-400, 1500, size=len(dates))
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df['Year'], df['Month'] = df['Date'].dt.year, df['Date'].dt.month
    return df

data = generate_data(selected_crop)

# 4. PREDICTION
if st.sidebar.button(f"Predict {selected_crop} Price"):
    model = RandomForestRegressor(n_estimators=100).fit(data[['Year', 'Month']], data['Price'])
    pred = model.predict([[target_date.year, target_date.month]])
    
    st.metric(label="Predicted Price (₹/Quintal)", value=f"₹{pred[0]:,.2f}")
    st.line_chart(data.set_index('Date')['Price'])