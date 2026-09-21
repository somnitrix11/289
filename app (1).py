
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the values for the delivery features to predict delay:')

# Input fields for each feature
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, value=19.35)
traffic_congestion = st.number_input('Traffic Congestion (1-5)', min_value=1, max_value=5, value=4, step=1)
weather_condition = st.number_input('Weather Condition (1-5)', min_value=1, max_value=5, value=3, step=1)
delivery_slot = st.number_input('Delivery Slot (1-3)', min_value=1, max_value=3, value=2, step=1)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, value=3)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=2)
road_condition_score = st.number_input('Road Condition Score (1-5)', min_value=1, max_value=5, value=4, step=1)
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, value=5.0)
fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, value=10.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (hours)', min_value=0, value=1)

# Create a DataFrame from the input values
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error('**Delivery is predicted to be Delayed!**')
    else:
        st.success('**Delivery is predicted to be On-Time!**')
    
    st.write(f"Probability of No Delay (Class 0): {prediction_proba[0]:.4f}")
    st.write(f"Probability of Delay (Class 1): {prediction_proba[1]:.4f}")

# Instructions to run the app
st.sidebar.markdown("""
## How to run this app:
1. Save the model as `logi.sav` in the same directory.
2. In your terminal, navigate to the directory where `app.py` is saved.
3. Run `streamlit run app.py`
""")
