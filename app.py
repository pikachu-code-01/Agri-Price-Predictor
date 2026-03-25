import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Agri Price Predictor", page_icon="🌾", layout="wide")

# CHANGED: Updated Title
st.title("🌾 Agri Price Predictor")
st.markdown("---")

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

# 2. SIDEBAR INPUTS
st.sidebar.header("Prediction Settings")
selected_crop = st.sidebar.selectbox("Choose a Crop", sorted(all_crops))
target_date = st.sidebar.date_input("Select Prediction Date", datetime.now())

# 3. DATA ENGINE (Logic updated for per-kg values)
@st.cache_data
def generate_market_data(crop_name):
    np.random.seed(len(crop_name)) 
    dates = pd.date_range(start='2023-01-01', end='2026-12-31', freq='MS')
    
    # CHANGED: Base price adjusted to look like per-kg (e.g., ₹20 to ₹150 per kg)
    base = (len(crop_name) * 2) + 20
    prices = base + np.random.randint(-10, 50, size=len(dates))
    
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

data = generate_market_data(selected_crop)

# 4. PREDICTION LOGIC
if st.sidebar.button(f"Predict {selected_crop} Price"):
    X = data[['Year', 'Month']]
    y = data['Price']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    pred_val = model.predict([[target_date.year, target_date.month]])
    
    # UI RESULTS
    col1, col2 = st.columns(2)
    with col1:
        # CHANGED: Label updated to Rupees per Kilogram
        st.metric(label=f"Predicted Market Price (₹/kg)", value=f"₹{pred_val[0]:,.2f}")
        st.write(f"Showing results for **{selected_crop}** on **{target_date.strftime('%B %Y')}**")
    
    with col2:
        avg_price = data['Price'].mean()
        if pred_val[0] > avg_price:
            st.warning("Trend: Price is expected to be above average.")
        else:
            st.success("Trend: Price is expected to be stable.")

    st.subheader("Price History & Trend Analysis (₹ per kg)")
    st.line_chart(data.set_index('Date')['Price'])
    
else:
    st.info("Select a crop and date from the sidebar, then click 'Predict'.")

st.markdown("---")
st.caption("Tech Stack: Python | Streamlit | Scikit-Learn | Pandas")