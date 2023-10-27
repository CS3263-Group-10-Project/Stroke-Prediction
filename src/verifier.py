import pickle
import pandas as pd
from pgmpy.inference import VariableElimination

with open("saved_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# 1. Prepare the Test Data
data = pd.read_csv('../data/transformed_data.csv')

# use the last 25% of the data for testing
test_data = data.iloc[int(len(data) * 0.80):].copy()
test_data.drop(columns=['stroke'], inplace=True)  # Assuming 'stroke' is what you want to predict

# 2. Make Predictions
inference = VariableElimination(loaded_model)

# Loop through each row in test_data and predict
predicted_values = []
for index, row in test_data.iterrows():
    prediction = inference.map_query(variables=['stroke'], evidence=row.to_dict())
    predicted_values.append(prediction['stroke'])

# 3. Evaluate Accuracy (remains the same)
actual_values = data.iloc[int(len(data) * 0.80):]['stroke'].tolist()
correct_predictions = sum([1 for pred, actual in zip(predicted_values, actual_values) if pred == actual])
accuracy = correct_predictions / len(actual_values) * 100

print(f"Model Accuracy: {accuracy:.2f}%")
