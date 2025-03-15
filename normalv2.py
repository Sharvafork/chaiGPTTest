import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ollama import chat, ChatResponse
import subprocess  

# Load CSV file
file_path = "cluster_0.csv"
data = pd.read_csv(file_path)

# Normalize column names (strip spaces, remove newlines, convert to lowercase)
data.columns = data.columns.str.strip().str.replace("\n", " ").str.lower()

# Print available columns for debugging
print("Available Columns:", data.columns.tolist())

# Identify categorical and numeric columns
categorical_cols = data.select_dtypes(include=['object']).columns  
numeric_cols = data.select_dtypes(include=['number']).columns  

# Handle missing values
num_imputer = SimpleImputer(strategy="mean")  # Fill missing numeric values
cat_imputer = SimpleImputer(strategy="most_frequent")  # Fill missing categorical values

data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Encode categorical data using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Convert categorical strings to numbers
    label_encoders[col] = le

# Ensure all data is numeric before clustering
df_cleaned = pd.DataFrame(data, columns=data.columns)

# Validate that no object columns remain
assert df_cleaned.select_dtypes(include=['object']).empty, "Some categorical columns were not properly converted."

# Print final DataFrame info for debugging
print(df_cleaned.info())

# Define max CPU usage
LOKY_MAX_CPU_COUNT = 4

# Generate response from Codellama model
response: ChatResponse = chat(model='codellama', messages=[
    {
        'role': 'user',
        'content': f"""# Only provide the code. Do not include any explanations.
# Any necessary details should be in comments.
# Ensure all necessary libraries are imported.

# Task: Perform clustering on 'cluster_0.csv' using sklearn.
# Generate a detailed PDF report that includes:
#  - Data overview
#  - Identified clusters
#  - Business insights and strategies based on clustering results.

# Ensure proper data preprocessing:
#  - Handle missing values to prevent ValueError: Input X contains NaN.
#  - Convert non-numeric values appropriately to avoid ValueError: could not convert string to float.
#  - Ensure train_test_split is correctly imported to avoid NameError: name 'train_test_split' is not defined.
#  - Include: from sklearn.model_selection import train_test_split.
#  - Ensure that the code runs without errors.

{df_cleaned}
""",
    },
])

# Save the generated code to a Python file
script_path = "cluster.py"


with open(script_path, "w") as f:  
    f.write(response.message.content.replace('```', '').replace('python', ''))

# Run the script using Conda environment (Python 3.10 in vmpy)
try:
    subprocess.run(["conda", "run", "--no-capture-output", "-n", "vmpy", "python", script_path], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running cluster.py: {e}")
    exit(1)

