import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

# Set MLflow tracking URI (you can change this to your preferred location)
mlflow.set_tracking_uri("file:./mlruns")

def load_and_preprocess_data(file_path):
    """Load and preprocess the student data"""
    df_students = pd.read_csv(file_path)
    return df_students

def perform_clustering(df, n_clusters_range=range(2, 15)):
    """Perform clustering with MLflow tracking"""
    # Start MLflow run
    with mlflow.start_run(run_name=f"clustering_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("n_clusters_range", list(n_clusters_range))
        
        # Scale the data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df),
            columns=df.columns
        )
        
        # Store results
        sse = {}
        silhouette_scores = {}
        
        # Try different numbers of clusters
        for k in n_clusters_range:
            print(f"Fitting model with {k} clusters")
            
            # Create and fit the model
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(df_scaled)
            
            # Calculate metrics
            sse[k] = kmeans.inertia_
            
            # Log metrics for this k
            mlflow.log_metric(f"sse_k{k}", kmeans.inertia_)
            
            # Log the model
            mlflow.sklearn.log_model(kmeans, f"kmeans_k{k}")
            
            # Log cluster sizes
            cluster_sizes = pd.Series(clusters).value_counts()
            for cluster, size in cluster_sizes.items():
                mlflow.log_metric(f"cluster_{cluster}_size_k{k}", size)
        
        # Log the elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(list(sse.keys()), list(sse.values()), 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('Elbow Method for Optimal k')
        plt.savefig('elbow_curve.png')
        mlflow.log_artifact('elbow_curve.png')
        
        return sse

def main():
    # Create directory for MLflow if it doesn't exist
    if not os.path.exists('mlruns'):
        os.makedirs('mlruns')
    
    # Load and preprocess data
    df_students = load_and_preprocess_data('data/df_students_num_imputed_cluster.csv')
    
    # Perform clustering with MLflow tracking
    sse = perform_clustering(df_students)
    
    # Print results
    print("\nClustering Results:")
    print("==================")
    for k, inertia in sse.items():
        print(f"k={k}: SSE={inertia:.2f}")

if __name__ == "__main__":
    main() 