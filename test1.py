import pandas as pd
import numpy as np
import os
import logging
import time
import subprocess
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ollama import chat, ChatResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
file_path = "cluster_0.csv"
data = pd.read_csv(file_path)

# Normalize column names
data.columns = data.columns.str.strip().str.replace("\n", " ").str.lower()

# Identify categorical and numeric columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

# Handle missing values
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Encode categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Use Ollama to generate KPIs
response: ChatResponse = chat(model="llama3.2", messages=[
{"role": "user", "content": "Given the following dataset summary, suggest the most relevant Key Performance Indicators (KPIs) for business insights:\n" + str(data.describe())}
])

ollama_response = response.message.content.strip()
logging.info(f"Ollama returned KPIs: {ollama_response}")

kpis = ollama_response.split(',')
kpis = [kpi.strip().lower() for kpi in kpis if kpi.strip().lower() in numeric_cols]

# Fallback if no valid KPIs are selected
default_kpis = numeric_cols[:5]  # Use first 5 numeric columns as a backup
if not kpis:
    logging.warning("No valid KPIs selected from Ollama response. Using default KPIs.")
    kpis = default_kpis

logging.info(f"Final KPIs used: {kpis}")

# Feature Selection
X = data[kpis]
y = np.random.randint(0, 2, size=len(X))  # Dummy target for feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=min(5, len(kpis)))
X_selected = selector.fit_transform(X, y)
selected_features = [kpis[i] for i in selector.get_support(indices=True)]
logging.info(f"Selected Features: {selected_features}")

# Clustering - Determine optimal clusters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[selected_features])
scores = {}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores[k] = score

optimal_k = max(scores, key=scores.get)
logging.info(f"Optimal number of clusters: {optimal_k}")

# Final Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

# Use Ollama to generate clustering report
report_response: ChatResponse = chat(model="deepseek-r1:8b", messages=[
    {"role": "user", "content": f"""Generate a detailed clustering report based on the following insights:

Optimal Clusters: {optimal_k}
Selected Features: {', '.join(selected_features)}

Dataset Overview:
{data.describe()}"""}
])

report_content = report_response.message.content

# Save Report
report_path = "clustering_report.txt"
with open(report_path, "w") as report_file:
    report_file.write(report_content)

logging.info("Clustering report saved as clustering_report.txt")
