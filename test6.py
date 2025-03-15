import pandas as pd
import numpy as np
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
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

def perform_clustering(data, kpis, dataset_name):
    """Clusters data based on KPIs and assigns meaningful cluster names."""
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
    logging.info(f"Optimal clusters for {dataset_name}: {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    data["cluster"] = kmeans.fit_predict(X_scaled)

    # Assign meaningful names to clusters
    cluster_names = {i: f"{dataset_name} - Cluster {i}" for i in range(optimal_k)}
    data["cluster_name"] = data["cluster"].map(cluster_names)
    
    return data, optimal_k, selected_features, cluster_names

def generate_visualizations(data, selected_features, dataset_name):
    """Generates enhanced visualizations for better understanding."""
    plots = []

    # Scatter Plot for Clusters
    if len(selected_features) >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data[selected_features[0]], y=data[selected_features[1]], hue=data["cluster_name"], palette="viridis")
        plt.title(f"Cluster Scatter Plot - {dataset_name}")
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1])
        plt.legend(title="Clusters", bbox_to_anchor=(1, 1))
        scatter_plot_path = f"{dataset_name}_scatter.png"
        plt.savefig(scatter_plot_path, bbox_inches="tight")
        plots.append(scatter_plot_path)
        plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data[selected_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Feature Correlation Heatmap - {dataset_name}")
    heatmap_path = f"{dataset_name}_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plots.append(heatmap_path)
    plt.close()

    # Bar Plot of Cluster Counts
    plt.figure(figsize=(8, 6))
    sns.countplot(x="cluster_name", data=data, palette="viridis")
    plt.title(f"Cluster Distribution - {dataset_name}")
    plt.xticks(rotation=45, ha="right")
    bar_plot_path = f"{dataset_name}_barplot.png"
    plt.savefig(bar_plot_path, bbox_inches="tight")
    plots.append(bar_plot_path)
    plt.close()

    return plots

def run_pipeline(file_paths, report_type):
    """Executes the clustering pipeline for individual or collective reports."""
    
    clustered_datasets = []
    reports = []
    visualizations = []
    
    for file_path in file_paths:
        dataset_name = os.path.basename(file_path).replace(".csv", "")
        logging.info(f"Processing {dataset_name}...")
        
        try:
            data, numeric_cols = process_dataset(file_path)
            kpis = generate_kpis(data, numeric_cols)
            clustered_data, optimal_k, selected_features, cluster_names = perform_clustering(data, kpis, dataset_name)
            clustered_datasets.append(clustered_data)
            visuals = generate_visualizations(clustered_data, selected_features, dataset_name)
            visualizations.extend(visuals)
            
            if report_type == "Individual Reports":
                report = f"### {dataset_name}\nClusters: {optimal_k}\nSelected Features: {selected_features}\n\nCluster Names: {cluster_names}\n"
                reports.append(report)
        except Exception as e:
            logging.error(f"Error processing {dataset_name}: {str(e)}")
            reports.append(f"Error processing {dataset_name}: {str(e)}\n\n")
    
    return "\n".join(reports), visualizations

def webui(file_paths, report_type):
    return run_pipeline(file_paths, report_type)

iface = gr.Interface(
    fn=webui,
    inputs=[
        gr.Files(label="Upload CSV Files", type="filepath"),
        gr.Radio(choices=["Individual Reports", "Collective Report"], label="Select Report Type")
    ],
    outputs=[gr.Textbox(label="Clustering Reports", lines=25), gr.Gallery(label="Visualizations")],
    title="KPI-Based Clustering Analysis",
    description="Upload CSV files and choose between generating individual reports or a collective analysis with meaningful visualizations."
)

if __name__ == "__main__":
    iface.launch()
