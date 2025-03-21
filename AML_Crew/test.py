import pandas as pd
import joblib

# Load the trained model and transformer
model = joblib.load('xgboost_model.joblib')
transformer = joblib.load('transformer.joblib')  # Load the saved transformer


sample_data = {
    'From Bank': [70],
    'Account': ['100428660'],
    'To Bank': [1124],
    'Account.1': ['800825340'],
    'Amount Received': [389769.39],
    'Receiving Currency': ['US Dollar'],
    'Amount Paid': [389769.39],
    'Payment Currency': ['US Dollar'],
    'Payment Format': ['Cheque'],
    'Date': ['2022-09-01'],
    'Day': ['Thursday'],
    'Time': ['00:21:00']
}


# Convert sample data to a DataFrame
sample_df = pd.DataFrame(sample_data)

# Predict using the trained model
prediction = model.predict(sample_df)

# Output the result
print(f"Prediction: {prediction[0]}")  # 'prediction[0]' will give the class (0 or 1)
