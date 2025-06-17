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
        f"The estimated price of the ticket is: **₹ {round(predicted_price[0]):,}**"
    )


st.header("About the Model", divider="gray")
st.subheader("Dataset Details")
st.markdown(
    """
The dataset used has been taken from [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/data).

- `airline`: The name of the airline company.
- `flight`: The flight code of the plane.
- `source_city`: The city from which the flight takes off.
- `departure_time`: The departure time of the flight.
- `stops`: The number of stops the flight makes en-route.
- `arrival_time`: The arrival time of the flight.
- `destination_city`: The city where the flight will land.
- `class`: The ticket class (_Economy_ or _Business_).
- `duration`: The time required to travel between the two cities.
- `days_left`: The number of days left for the flight/trip.
- `price`: The ticket price (in _INR_).
"""
)

st.subheader("Model Performance")

st.write(""" 
    The model used is a **Random Forest Regressor**, with hyperparameters tuned. Training was performed on the above dataset, which consists of approximately 300261 distinct flight booking options. The model achieved the following performance metrics on the training and validation/test datasets:
""")

metrics = pd.DataFrame(
    {
        "Data": ["Training Data", "Validation/Test Data"],
        "Mean Absolute Error (MAE)": [2333, 2344],
        "R-squared (R²)": [0.965, 0.964],
        "Mean Squared Error (MSE)(~)": [1.82e+07, 1.83e+07],
    }
)

st.table(metrics)

st.divider()
