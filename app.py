import streamlit as st
import pandas as pd
import joblib

model = joblib.load("flight_price_model.pkl")
model_columns = joblib.load("flight_price_model_columns.pkl")

st.title("Flight Price Predictor")

st.subheader("Estimate the price of domestic flight tickets", divider="gray")

st.write(
    """
    A machine learning model which predicts the prices of domestic flights to and from any of the metro cities based on various parameters like airline, number of stops/layovers, arrival and departure times, origin, destination, ticket class, and the number of days from the trip.     
"""
)

st.sidebar.header("Input Parameters")


def input_features():
    airline = st.sidebar.selectbox(
        "Airline", ["AirAsia", "Air India", "GO FIRST", "Indigo", "SpiceJet", "Vistara"]
    )
    stops = st.sidebar.selectbox("Number of Stops", ["zero", "one", "two or more"])
    departure_time = st.sidebar.selectbox(
        "Departure Time",
        ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"],
    )
    arrival_time = st.sidebar.selectbox(
        "Arrival Time",
        ["Late Night", "Night", "Evening", "Afternoon", "Morning", "Early Morning"],
    )
    origin = st.sidebar.selectbox(
        "Origin", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Kolkata"]
    )
    destination = st.sidebar.selectbox(
        "Destination",
        ["Kolkata", "Mumbai", "Hyderabad", "Delhi", "Chennai", "Bangalore"],
    )
    ticket_class = st.sidebar.selectbox("Ticket Class", ["Economy", "Business"])
    days_from_trip = st.sidebar.slider(
        "Days from Trip", min_value=0, max_value=49, value=20
    )

    if origin == destination:
        st.error("Origin and Destination cannot be the same.")
        return pd.DataFrame()
    
    if departure_time == arrival_time:
        st.error("Departure time and Arrival time cannot be the same.")
        return pd.DataFrame()

    # features = ['airline', 'stops', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class', 'days_left']

    features = pd.DataFrame(
        [
            {
                "airline": airline.replace(" ", "_"),
                "stops": stops.replace(" ", "_"),
                "source_city": origin,
                "departure_time": departure_time,
                "arrival_time": arrival_time.replace(" ", "_"),
                "destination_city": destination,
                "class": ticket_class,
                "days_left": days_from_trip,
            }
        ]
    )
    features_encoded = pd.get_dummies(features)
    features_encoded = features_encoded.reindex(columns=model_columns, fill_value=0)

    return features_encoded


input = input_features()

disable = input.empty


if st.button("Predict", disabled=disable):
    predicted_price = model.predict(input)
    st.success(
        f"The estimated price of the ticket is: {round(predicted_price[0]):,} INR"
    )

st.divider()
