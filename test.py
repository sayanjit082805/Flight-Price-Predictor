# features = ['airline', 'stops', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class', 'days_left']

import joblib
import pandas as pd

model = joblib.load("flight_price_model.pkl")
model_columns = joblib.load("flight_price_model_columns.pkl")

new_price = pd.DataFrame(
    [
        {
            "airline": "Air_India",
            "stops": "zero",
            "source_city": "Kolkata",
            "departure_time": "Morning",
            "arrival_time": "Afternoon",
            "destination_city": "Mumbai",
            "class": "Economy",
            "days_left": 5,
        }
    ]
)

new_price_encoded = pd.get_dummies(new_price)
new_price_encoded = new_price_encoded.reindex(columns=model_columns, fill_value=0)

predicted_price = model.predict(new_price_encoded)
print(f"The price of the ticket is: {round(predicted_price[0]):,} INR")
