import logging
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
data = pd.read_csv('../data/transformed_data.csv')

# Drop the id column
data.drop(columns=['id'], inplace=True)

# Use only the first 100 rows for training
data_subset = data.iloc[:100]

# Adjust the edges to match dataset columns, removing references to "Lifestyle"
edges = [
    ("age", "hypertension"),
    ("age", "heart_disease"),
    ("age", "stroke"),
    ("gender", "smoking_status"),
    ("gender", "stroke"),
    ("hypertension", "stroke"),
    ("heart_disease", "stroke"),
    ("ever_married", "stroke"),
    ("isPrivate", "stroke"),
    ("isSelfEmployed", "stroke"),
    ("avg_glucose_level", "stroke"),
    ("bmi", "hypertension"),
    ("bmi", "heart_disease"),
    ("bmi", "stroke"),
    ("smoking_status", "hypertension"),
    ("smoking_status", "heart_disease"),
    ("smoking_status", "stroke")
]

# Define the Bayesian model structure based on the provided edges
model = BayesianNetwork(edges)

# Train the model using Maximum Likelihood Estimators on the subset data
model.fit(data_subset, estimator=MaximumLikelihoodEstimator)

# Display the learned CPDs for a sample node (e.g., "stroke")
print(model.get_cpds("stroke"))
