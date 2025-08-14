import pandas as pd

def assign_segment_labels(df_with_clusters, cluster_col='segment'):
    """Assigns business-friendly labels to numeric cluster IDs."""
    
    # Calculate centroids to understand cluster characteristics
    centroids = df_with_clusters.groupby(cluster_col).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean',
        'probabilistic_clv_90d': 'mean'
    }).reset_index()

    # Order centroids by CLV to assign labels logically
    # Higher CLV -> 'Champion', Lower Recency -> Better
    centroids = centroids.sort_values(by='probabilistic_clv_90d', ascending=False)
    
    if len(centroids) == 3:
        labels = ["High-Value Champion", "Potential Loyalist", "At-Risk/New"]
    elif len(centroids) == 4:
         labels = ["High-Value Champion", "Potential Loyalist", "Needs Attention", "At-Risk/New"]
    else: # Default for other k values
        labels = [f"Segment {i}" for i in range(len(centroids))]

    label_map = {row[cluster_col]: labels[i] for i, row in centroids.iterrows()}
    
    df_with_clusters['segment_label'] = df_with_clusters[cluster_col].map(label_map)
    
    return df_with_clusters, label_map