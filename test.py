# features = ['airline', 'stops', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class', 'days_left']

import joblib
import pandas as pd

model = joblib.load("flight_price_model.pkl")
model_columns = joblib.load("flight_price_model_columns.pkl")

new_price = pd.DataFrame(
    [
        {
            "airline": "AirAsia",
            "stops": "zero",
            "source_city": "Bangalore",
            "departure_time": "Early_Morning",
            "arrival_time": "Afternoon",
            "destination_city": "Kolkata",
            "class": "Economy",
            "days_left": 20,
        }
    ]
)

new_price_encoded = pd.get_dummies(new_price)
new_price_encoded = new_price_encoded.reindex(columns=model_columns, fill_value=0)

predicted_price = model.predict(new_price_encoded)
print(f"The price of the ticket is: {round(predicted_price[0]):,} INR")
