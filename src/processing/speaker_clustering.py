"""
Speaker clustering module for grouping speaker embeddings
Provides multiple clustering algorithms with configurable parameters
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

from src.core.config import CONFIG

logger = logging.getLogger(__name__)

class SpeakerClustering:
    """
    Speaker clustering using various algorithms
    """
    
    def __init__(self, algorithm='agglomerative'):
        self.algorithm = algorithm
        self.min_speakers = CONFIG.get('min_speakers', 1)
        self.max_speakers = CONFIG.get('max_speakers', 10)
        
        # Clustering parameters from config
        self.cluster_threshold = CONFIG.get('cluster_threshold', 0.3)
        self.min_cluster_size = CONFIG.get('min_cluster_size', 2)
        self.linkage_method = CONFIG.get('clustering_linkage', 'ward')
        self.affinity = CONFIG.get('clustering_affinity', 'euclidean')
        
        # DBSCAN parameters
        self.dbscan_eps = CONFIG.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = CONFIG.get('dbscan_min_samples', 2)
        
        # Adaptive clustering
        self.enable_adaptive_clustering = CONFIG.get('enable_adaptive_clustering', True)
        self.silhouette_threshold = CONFIG.get('silhouette_threshold', 0.3)
        
        logger.info(f"Initialized SpeakerClustering with algorithm: {algorithm}")
    
    def cluster_embeddings(self, embeddings, timestamps=None):
        """
        Cluster speaker embeddings into speaker groups
        
        Args:
            embeddings: numpy array of embeddings (n_segments, embedding_dim)
            timestamps: Optional list of timestamps for each embedding
            
        Returns:
            dict: Clustering results with labels and metrics
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings provided for clustering")
            return {
                'labels': [],
                'n_clusters': 0,
                'silhouette_score': 0.0,
                'algorithm': self.algorithm
            }
        
        if len(embeddings) == 1:
            logger.info("Single embedding provided, assigning to cluster 0")
            return {
                'labels': [0],
                'n_clusters': 1,
                'silhouette_score': 1.0,
                'algorithm': self.algorithm
            }
        
        logger.info(f"Clustering {len(embeddings)} embeddings using {self.algorithm}")
        
        try:
            if self.algorithm == 'agglomerative':
                return self._agglomerative_clustering(embeddings, timestamps)
            elif self.algorithm == 'dbscan':
                return self._dbscan_clustering(embeddings, timestamps)
            else:
                raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")
                
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            # Fallback: assign all to single cluster
            return {
                'labels': [0] * len(embeddings),
                'n_clusters': 1,
                'silhouette_score': 0.0,
                'algorithm': self.algorithm,
                'error': str(e)
            }
    
    def _agglomerative_clustering(self, embeddings, timestamps=None):
        """Perform agglomerative clustering"""
        
        # Determine optimal number of clusters
        n_clusters = self._determine_optimal_clusters(embeddings)
        
        # Perform clustering
        if self.affinity == 'cosine':
            # Use cosine distance for speaker embeddings
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average',  # Ward doesn't work with cosine
                metric='cosine'
            )
        else:
            # Use euclidean distance with ward linkage
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.linkage_method,
                metric=self.affinity
            )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = clusterer.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette = self._calculate_silhouette_score(embeddings, labels)
        
        # Post-process labels to ensure they're contiguous
        labels = self._relabel_clusters(labels)
        
        result = {
            'labels': labels.tolist(),
            'n_clusters': len(np.unique(labels)),
            'silhouette_score': silhouette,
            'algorithm': 'agglomerative'
        }
        
        logger.info(f"Agglomerative clustering completed: {result['n_clusters']} clusters, "
                   f"silhouette score: {result['silhouette_score']:.3f}")
        
        return result
    
    def _dbscan_clustering(self, embeddings, timestamps=None):
        """Perform DBSCAN clustering"""
        
        # Normalize embeddings for DBSCAN
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Perform DBSCAN
        clusterer = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric='cosine'
        )
        
        labels = clusterer.fit_predict(normalized_embeddings)
        
        # Handle noise points (labeled as -1)
        noise_mask = labels == -1
        if np.any(noise_mask):
            # Assign noise points to nearest cluster or create new clusters
            labels = self._handle_noise_points(labels, normalized_embeddings)
        
        # Calculate silhouette score
        silhouette = self._calculate_silhouette_score(normalized_embeddings, labels)
        
        # Post-process labels
        labels = self._relabel_clusters(labels)
        
        result = {
            'labels': labels.tolist(),
            'n_clusters': len(np.unique(labels)),
            'silhouette_score': silhouette,
            'algorithm': 'dbscan'
        }
        
        logger.info(f"DBSCAN clustering completed: {result['n_clusters']} clusters, "
                   f"silhouette score: {result['silhouette_score']:.3f}")
        
        return result
    
    def _determine_optimal_clusters(self, embeddings):
        """Determine optimal number of clusters using multiple criteria"""
        
        n_samples = len(embeddings)
        
        # Constrain by min/max speakers
        min_clusters = max(1, self.min_speakers)
        max_clusters = min(n_samples, self.max_speakers)
        
        if min_clusters == max_clusters:
            return min_clusters
        
        if not self.enable_adaptive_clustering:
            return min_clusters
        
        # Try different numbers of clusters and evaluate
        best_score = -1
        best_n_clusters = min_clusters
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            try:
                # Quick clustering to evaluate
                if self.affinity == 'cosine':
                    clusterer = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage='average',
                        metric='cosine'
                    )
                else:
                    clusterer = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage=self.linkage_method,
                        metric=self.affinity
                    )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = clusterer.fit_predict(embeddings)
                
                # Calculate silhouette score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                
            except Exception as e:
                logger.debug(f"Failed to evaluate {n_clusters} clusters: {e}")
                continue
        
        logger.info(f"Optimal number of clusters: {best_n_clusters} (score: {best_score:.3f})")
        return best_n_clusters
    
    def _calculate_silhouette_score(self, embeddings, labels):
        """Calculate silhouette score for clustering quality"""
        try:
            if len(np.unique(labels)) <= 1:
                return 0.0
            
            return silhouette_score(embeddings, labels)
            
        except Exception as e:
            logger.debug(f"Failed to calculate silhouette score: {e}")
            return 0.0
    
    def _handle_noise_points(self, labels, embeddings):
        """Handle noise points from DBSCAN by assigning to nearest cluster"""
        
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        # Find cluster centers
        unique_labels = np.unique(labels[~noise_mask])
        if len(unique_labels) == 0:
            # All points are noise, assign to single cluster
            return np.zeros_like(labels)
        
        cluster_centers = {}
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_centers[label] = np.mean(embeddings[cluster_mask], axis=0)
        
        # Assign noise points to nearest cluster
        for i, is_noise in enumerate(noise_mask):
            if is_noise:
                # Find nearest cluster
                min_dist = float('inf')
                nearest_cluster = unique_labels[0]
                
                for label, center in cluster_centers.items():
                    dist = np.linalg.norm(embeddings[i] - center)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_cluster = label
                
                labels[i] = nearest_cluster
        
        return labels
    
    def _relabel_clusters(self, labels):
        """Relabel clusters to be contiguous starting from 0"""
        unique_labels = np.unique(labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        relabeled = np.array([label_map[label] for label in labels])
        return relabeled
    
    def update_parameters(self, **kwargs):
        """Update clustering parameters"""
        
        updated_params = []
        
        if 'min_speakers' in kwargs:
            self.min_speakers = kwargs['min_speakers']
            updated_params.append('min_speakers')
        
        if 'max_speakers' in kwargs:
            self.max_speakers = kwargs['max_speakers']
            updated_params.append('max_speakers')
        
        if 'cluster_threshold' in kwargs:
            self.cluster_threshold = kwargs['cluster_threshold']
            updated_params.append('cluster_threshold')
        
        if 'algorithm' in kwargs:
            self.algorithm = kwargs['algorithm']
            updated_params.append('algorithm')
        
        if updated_params:
            logger.info(f"Updated clustering parameters: {updated_params}")
    
    def get_cluster_statistics(self, embeddings, labels):
        """Get statistics about the clustering results"""
        
        if len(embeddings) == 0 or len(labels) == 0:
            return {}
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        stats = {
            'n_clusters': n_clusters,
            'cluster_sizes': {},
            'cluster_compactness': {},
            'inter_cluster_distances': {}
        }
        
        # Calculate cluster sizes and compactness
        for label in unique_labels:
            cluster_mask = np.array(labels) == label
            cluster_embeddings = embeddings[cluster_mask]
            
            stats['cluster_sizes'][int(label)] = len(cluster_embeddings)
            
            if len(cluster_embeddings) > 1:
                # Calculate average intra-cluster distance
                distances = pdist(cluster_embeddings, metric='cosine')
                stats['cluster_compactness'][int(label)] = np.mean(distances)
            else:
                stats['cluster_compactness'][int(label)] = 0.0
        
        # Calculate inter-cluster distances
        if n_clusters > 1:
            cluster_centers = {}
            for label in unique_labels:
                cluster_mask = np.array(labels) == label
                cluster_centers[label] = np.mean(embeddings[cluster_mask], axis=0)
            
            for i, label1 in enumerate(unique_labels):
                for label2 in unique_labels[i+1:]:
                    dist = np.linalg.norm(cluster_centers[label1] - cluster_centers[label2])
                    stats['inter_cluster_distances'][f'{int(label1)}-{int(label2)}'] = dist
        
        return stats 