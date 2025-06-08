import mlflow
import mlflow.sklearn
from datetime import datetime
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA

def monitor_clustering(df, df_scaled, clustering_func, n_clusters_range=range(2, 15), 
                      experiment_name="clustering_experiment", save_dir="data/processed/clustering_results"):
    """
    Automatically monitor clustering process with MLflow.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe with features
    df_scaled : pandas.DataFrame
        Scaled dataframe used for clustering
    clustering_func : function
        Function that performs clustering. Should accept n_clusters and return (model, clusters)
    n_clusters_range : range
        Range of cluster numbers to try
    experiment_name : str
        Name for the MLflow experiment
    save_dir : str
        Directory to save results
    """
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    def plot_cluster_characteristics(df, clusters, features, k):
        """Create and save plots showing cluster characteristics"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Feature distributions by cluster
        plt.subplot(2, 2, 1)
        for feature in features[:4]:  # Plot first 4 features
            sns.boxplot(x=clusters, y=df[feature])
        plt.title(f'Feature Distributions by Cluster (k={k})')
        plt.xticks(rotation=45)
        
        # Plot 2: Cluster sizes
        plt.subplot(2, 2, 2)
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
        plt.title(f'Cluster Sizes (k={k})')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Students')
        
        # Plot 3: Feature correlations
        plt.subplot(2, 2, 3)
        corr_matrix = df[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        
        # Plot 4: PCA visualization
        plt.subplot(2, 2, 4)
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df[features])
        plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis')
        plt.title(f'PCA Visualization (k={k})')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        plt.tight_layout()
        plt.savefig(f'cluster_analysis_k{k}.png')
        return f'cluster_analysis_k{k}.png'

    # Main clustering experiment
    with mlflow.start_run(run_name=f"clustering_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("n_clusters_range", list(n_clusters_range))
        mlflow.log_param("features_used", list(df_scaled.columns))
        
        sse = {}
        silhouette_scores = {}
        calinski_scores = {}
        
        for k in n_clusters_range:
            print(f"Fitting with {k} clusters")
            
            # Get model and clusters from the provided clustering function
            model, clusters = clustering_func(k)
            
            # Calculate metrics
            sse[k] = model.inertia_
            silhouette_scores[k] = silhouette_score(df_scaled, clusters)
            calinski_scores[k] = calinski_harabasz_score(df_scaled, clusters)
            
            # Log metrics
            mlflow.log_metric(f"sse_k{k}", model.inertia_)
            mlflow.log_metric(f"silhouette_k{k}", silhouette_scores[k])
            mlflow.log_metric(f"calinski_k{k}", calinski_scores[k])
            
            # Log the model
            mlflow.sklearn.log_model(model, f"kmeans_k{k}")
            
            # Log cluster sizes
            cluster_sizes = pd.Series(clusters).value_counts()
            for cluster, size in cluster_sizes.items():
                mlflow.log_metric(f"cluster_{cluster}_size_k{k}", size)
            
            # Create and log cluster analysis plots
            plot_path = plot_cluster_characteristics(df, clusters, df_scaled.columns, k)
            mlflow.log_artifact(plot_path)
        
        # Log the evaluation curves
        plt.figure(figsize=(15, 5))
        
        # Elbow curve
        plt.subplot(1, 3, 1)
        plt.plot(list(sse.keys()), list(sse.values()), 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('Elbow Method')
        
        # Silhouette score curve
        plt.subplot(1, 3, 2)
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), 'ro-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        
        # Calinski-Harabasz score curve
        plt.subplot(1, 3, 3)
        plt.plot(list(calinski_scores.keys()), list(calinski_scores.values()), 'go-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Calinski-Harabasz Score')
        plt.title('Calinski-Harabasz Analysis')
        
        plt.tight_layout()
        plt.savefig('evaluation_curves.png')
        mlflow.log_artifact('evaluation_curves.png')
        
        # Find optimal k based on silhouette score
        optimal_k = max(silhouette_scores.items(), key=lambda x: x[1])[0]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        
        return optimal_k, sse, silhouette_scores, calinski_scores

def save_clustering_results(df, df_scaled, model, clusters, save_dir="data/processed/clustering_results"):
    """Save clustering results and log them to MLflow"""
    
    with mlflow.start_run(run_name=f"save_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Create directory for results if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the model
        model_path = os.path.join(save_dir, 'kmeans_model.joblib')
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Save clustering parameters
        clustering_params = {
            'n_clusters': model.n_clusters,
            'random_state': model.random_state,
            'n_init': model.n_init,
            'max_iter': model.max_iter,
            'algorithm': model.algorithm,
            'variables_used': list(df_scaled.columns),
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'sse': float(model.inertia_),
                'silhouette_score': float(silhouette_score(df_scaled, clusters)),
                'calinski_harabasz_score': float(calinski_harabasz_score(df_scaled, clusters))
            }
        }
        
        params_path = os.path.join(save_dir, 'clustering_params.json')
        with open(params_path, 'w') as f:
            json.dump(clustering_params, f, indent=4)
        mlflow.log_artifact(params_path)
        
        # Save the clustered data
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clusters
        data_path = os.path.join(save_dir, 'df_students_clustered.csv')
        df_with_clusters.to_csv(data_path, index=False)
        mlflow.log_artifact(data_path)
        
        return df_with_clusters 