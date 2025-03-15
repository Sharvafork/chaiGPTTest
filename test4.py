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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_dataset(file_path):
    """Loads, cleans, and preprocesses a dataset."""
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip().str.replace("\n", " ").str.lower()
    
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    
    data.fillna(0, inplace=True)  # Ensure no NaN values remain
    
    return data, numeric_cols

def generate_kpis(data, numeric_cols):
    """Uses LLM to determine relevant KPIs."""
    response = chat(model="llama3.2", messages=[
        {"role": "user", "content": f"Given the dataset summary, suggest the most relevant KPIs:\n\n{data.describe()}"}
    ])
    
    ollama_response = response.message.content.strip()
    logging.info(f"Ollama returned KPIs: {ollama_response}")
    
    kpis = ollama_response.split(",")
    kpis = [kpi.strip().lower() for kpi in kpis if kpi.strip().lower() in numeric_cols]
    
    return kpis if kpis else numeric_cols[:5]

def perform_clustering(data, kpis):
    """Clusters data based on KPIs."""
    X = data[kpis]
    
    # Handle missing values if any remain
    if X.isnull().sum().sum() > 0:
        logging.warning("NaN values found after preprocessing. Filling with mean values.")
        X.fillna(X.mean(), inplace=True)

    selector = SelectKBest(score_func=mutual_info_classif, k=min(5, len(kpis)))
    X_selected = selector.fit_transform(X, np.random.randint(0, 2, size=len(X)))
    selected_features = [kpis[i] for i in selector.get_support(indices=True)]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[selected_features])
    
    scores = {k: silhouette_score(X_scaled, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled))
              for k in range(2, 10)}
    
    optimal_k = max(scores, key=scores.get)
    logging.info(f"Optimal clusters: {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    data["cluster"] = kmeans.fit_predict(X_scaled)
    
    return data, optimal_k, selected_features

def merge_datasets(file_paths):
    """Merges multiple datasets for a collective analysis."""
    combined_data = []
    for file_path in file_paths:
        data, _ = process_dataset(file_path)
        data["source_file"] = os.path.basename(file_path)  # Track original dataset
        combined_data.append(data)
    
    return pd.concat(combined_data, ignore_index=True)

def generate_combined_report(clustered_datasets):
    """Generates a report combining all datasets' clustering results."""
    
    # Combine clustering data from all datasets
    combined_df = pd.concat(clustered_datasets, ignore_index=True)
    correlation_matrix = combined_df.corr().to_dict()
    
    # Pass all clustering information to Ollama
    response = chat(model="deepseek-r1:8b", messages=[
        {"role": "user", "content": f"Analyze the following clustering results from multiple datasets:\n\n"
        f"## Clustering Summary\n"
        f"- Number of datasets: {len(clustered_datasets)}\n"
        f"- Total entries analyzed: {len(combined_df)}\n"
        f"- Key features used for clustering: {', '.join(combined_df.columns[:5])}\n\n"
        f"## Cluster Insights\n"
        f"{combined_df.groupby('cluster').mean().to_dict()}\n\n"
        f"## Correlation Across Datasets\n"
        f"{correlation_matrix}\n\n"
        f"## Business Recommendations\n"
        f"- How do the clusters relate to each other?\n"
        f"- What trends or outliers are present?\n"
        f"- What actions should businesses take based on these findings?\n\n"
        f"## Conclusion\n"
        f"Provide an executive summary based on these multi-dataset insights."}
    ])
    
    return response.message.content

def run_pipeline(file_paths, report_type):
    """Executes the clustering pipeline for individual or collective reports."""
    
    clustered_datasets = []
    reports = []
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        logging.info(f"Processing {file_name}...")
        
        try:
            data, numeric_cols = process_dataset(file_path)
            kpis = generate_kpis(data, numeric_cols)
            clustered_data, optimal_k, selected_features = perform_clustering(data, kpis)
            clustered_datasets.append(clustered_data)
            
            if report_type == "Individual Reports":
                report = f"### {file_name}\nClusters: {optimal_k}\nSelected Features: {selected_features}\n\n"
                reports.append(report)
        except Exception as e:
            logging.error(f"Error processing {file_name}: {str(e)}")
            reports.append(f"Error processing {file_name}: {str(e)}\n\n")
    
    if report_type == "Collective Report":
        return generate_combined_report(clustered_datasets)
    else:
        return "\n".join(reports)

def webui(file_paths, report_type):
    return run_pipeline(file_paths, report_type)

iface = gr.Interface(
    fn=webui,
    inputs=[
        gr.Files(label="Upload CSV Files", type="filepath"),
        gr.Radio(choices=["Individual Reports", "Collective Report"], label="Select Report Type")
    ],
    outputs=gr.Textbox(label="Clustering Reports", lines=25),
    title="KPI-Based Clustering Analysis",
    description="Upload CSV files and choose between generating individual reports or a collective analysis."
)

if __name__ == "__main__":
    iface.launch()
