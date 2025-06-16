# Flight-Price-Predictor

A machine learning model which predicts the prices of domestic flights to and from any of the metro cities based on various parameters like airline, number of stops/layovers, arrival and departure times, origin, destination, ticket class, and the number of days from the trip.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

# Overview

This project implements a prediction system which can be used to predict the prices of flights based on various parameters. The machine leaning model analyses historical data to estimate flight prices.

# Features

- **Input Parameters**: airline, number of stops/layovers, arrival and departure times, origin, destination, ticket class, and the number of days from the trip.
- **Model**: Random Forest (Regressor)
- **Price Estimation**: Estimates the cost of domestic flights.

# Dataset

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

# Installation

1. Clone the repository:

```bash
git clone https://github.com/sayanjit082805/Flight-Price-Predictor.git
cd Flight-Price-Predictor
```

2. Create a virtual environment (recommended):

```bash
python -m venv flight_predictor_env
source flight_predictor_env/bin/activate  # On Windows: flight_predictor_env\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

# Usage

As of now, there are no deployments/explicit-frontend to run the model. A sample `test.py` has been provided, simply tweak the required parameters and run the file using `python3 test.py`.
(A frontend interface is planned)

```python
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
```

#### Sample Output

`The price of the ticket is: 8,191 INR`

# Model Performance

| Data          | MAE   | R² Score |
| ------------- | ----- | -------- |
| Training Data | 2333  | 0.965    |
| Test Data     | 2,344 | 0.964    |

# Acknowledgements

The dataset has been taken from kaggle.

# License

This project is licensed under The Unlicense License.
