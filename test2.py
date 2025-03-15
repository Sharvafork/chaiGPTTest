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
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_dataset(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip().str.replace("\n", " ").str.lower()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    
    return data, numeric_cols

def generate_kpis(data, numeric_cols):
    response = chat(model="llama3.2", messages=[
        {"role": "user", "content": f"""Given the following dataset summary, suggest the most relevant Key Performance Indicators (KPIs) for business insights:
        \n""" + str(data.describe())}
    ])
    
    ollama_response = response.message.content.strip()
    logging.info(f"Ollama returned KPIs: {ollama_response}")
    
    kpis = ollama_response.split(',')
    kpis = [kpi.strip().lower() for kpi in kpis if kpi.strip().lower() in numeric_cols]
    
    default_kpis = numeric_cols[:5]
    if not kpis:
        logging.warning("No valid KPIs selected from Ollama response. Using default KPIs.")
        kpis = default_kpis
    
    return kpis

def perform_clustering(data, kpis):
    X = data[kpis]
    selector = SelectKBest(score_func=mutual_info_classif, k=min(5, len(kpis)))
    X_selected = selector.fit_transform(X, np.random.randint(0, 2, size=len(X)))
    selected_features = [kpis[i] for i in selector.get_support(indices=True)]
    
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
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_scaled)
    
    return data, optimal_k, selected_features

def generate_report(data, optimal_k, selected_features):
    response = chat(model="deepseek-r1:8b", messages=[
        {"role": "user", "content": f"""Generate a detailed clustering report based on the following insights:
        \nOptimal Clusters: {optimal_k}\nSelected Features: {', '.join(selected_features)}\n\nDataset Overview:\n{data.describe()}"""}
    ])
    return response.message.content

def run_pipeline(file_path):
    data, numeric_cols = process_dataset(file_path)
    kpis = generate_kpis(data, numeric_cols)
    data, optimal_k, selected_features = perform_clustering(data, kpis)
    report = generate_report(data, optimal_k, selected_features)
    return report

def webui(file):
    report = run_pipeline(file.name)
    return report

iface = gr.Interface(
    fn=webui,
    inputs=gr.File(label="Upload CSV File"),
    outputs=gr.Textbox(label="Clustering Report"),
    title="KPI-Based Clustering Analysis",
    description="Upload a CSV file, and the system will generate a KPI-based clustering analysis with a formatted report."
)

if __name__ == "__main__":
    iface.launch()
