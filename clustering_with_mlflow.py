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

# Set up MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
if not os.path.exists('mlruns'):
    os.makedirs('mlruns')

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
    
    # Plot 4: PCA visualization (if more than 2 features)
    if len(features) > 2:
        plt.subplot(2, 2, 4)
        from sklearn.decomposition import PCA
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
    mlflow.log_param("n_clusters_range", list(range(2, 15)))
    mlflow.log_param("features_used", list(df_scaled.columns))
    
    sse = {}
    silhouette_scores = {}
    calinski_scores = {}
    
    for k in range(2, 15):
        print(f"Fitting pipe with {k} clusters")
        
        clustering_model = KMeans(n_clusters=k, random_state=42)
        clusters = clustering_model.fit_predict(df_scaled)
        
        # Calculate metrics
        sse[k] = clustering_model.inertia_
        silhouette_scores[k] = silhouette_score(df_scaled, clusters)
        calinski_scores[k] = calinski_harabasz_score(df_scaled, clusters)
        
        # Log metrics
        mlflow.log_metric(f"sse_k{k}", clustering_model.inertia_)
        mlflow.log_metric(f"silhouette_k{k}", silhouette_scores[k])
        mlflow.log_metric(f"calinski_k{k}", calinski_scores[k])
        
        # Log the model
        mlflow.sklearn.log_model(clustering_model, f"kmeans_k{k}")
        
        # Log cluster sizes
        cluster_sizes = pd.Series(clusters).value_counts()
        for cluster, size in cluster_sizes.items():
            mlflow.log_metric(f"cluster_{cluster}_size_k{k}", size)
        
        # Create and log cluster analysis plots
        plot_path = plot_cluster_characteristics(df_students_feat, clusters, df_scaled.columns, k)
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

# Final clustering with optimal k
with mlflow.start_run(run_name=f"final_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    n_clusters = 5  # your chosen number of clusters
    
    # Log parameters
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("features_used", list(df_scaled.columns))
    
    # Create and fit the final model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Calculate and log metrics
    mlflow.log_metric("final_sse", kmeans.inertia_)
    mlflow.log_metric("final_silhouette", silhouette_score(df_scaled, clusters))
    mlflow.log_metric("final_calinski", calinski_harabasz_score(df_scaled, clusters))
    
    # Log cluster sizes and characteristics
    cluster_sizes = pd.Series(clusters).value_counts()
    for cluster, size in cluster_sizes.items():
        mlflow.log_metric(f"final_cluster_{cluster}_size", size)
        
        # Log cluster statistics
        cluster_data = df_students_feat[clusters == cluster]
        for feature in df_scaled.columns:
            mlflow.log_metric(f"cluster_{cluster}_{feature}_mean", cluster_data[feature].mean())
            mlflow.log_metric(f"cluster_{cluster}_{feature}_std", cluster_data[feature].std())
    
    # Log the final model
    mlflow.sklearn.log_model(kmeans, "final_kmeans_model")
    
    # Create and log final cluster analysis
    plot_path = plot_cluster_characteristics(df_students_feat, clusters, df_scaled.columns, n_clusters)
    mlflow.log_artifact(plot_path)
    
    # Add clusters to the original dataframe
    df_students_feat['cluster'] = clusters

# Save results
with mlflow.start_run(run_name=f"save_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    # Create directory for results if it doesn't exist
    results_dir = 'data/processed/clustering_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Save the model
    model_path = os.path.join(results_dir, 'kmeans_model.joblib')
    joblib.dump(kmeans, model_path)
    mlflow.log_artifact(model_path)
    
    # Save clustering parameters
    clustering_params = {
        'n_clusters': kmeans.n_clusters,
        'random_state': kmeans.random_state,
        'n_init': kmeans.n_init,
        'max_iter': kmeans.max_iter,
        'algorithm': kmeans.algorithm,
        'variables_used': list(df_scaled.columns),
        'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'sse': float(kmeans.inertia_),
            'silhouette_score': float(silhouette_score(df_scaled, clusters)),
            'calinski_harabasz_score': float(calinski_harabasz_score(df_scaled, clusters))
        }
    }
    
    params_path = os.path.join(results_dir, 'clustering_params.json')
    with open(params_path, 'w') as f:
        json.dump(clustering_params, f, indent=4)
    mlflow.log_artifact(params_path)
    
    # Save the clustered data
    data_path = os.path.join(results_dir, 'df_students_clustered_well.csv')
    df_students_feat.to_csv(data_path, index=False)
    mlflow.log_artifact(data_path) 