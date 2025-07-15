"""
Speaker clustering module for grouping speaker embeddings
Provides multiple clustering algorithms with configurable parameters
"""

import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
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
        self.cluster_threshold = CONFIG.get('cluster_threshold', 0.3)
        self.linkage_method = CONFIG.get('clustering_linkage', 'ward')
        self.affinity = CONFIG.get('clustering_affinity', 'euclidean')
        self.dbscan_eps = CONFIG.get('dbscan_eps', 0.5)
        self.dbscan_min_samples = CONFIG.get('dbscan_min_samples', 2)
        self.enable_adaptive_clustering = CONFIG.get('enable_adaptive_clustering', True)
        logger.info(f"Initialized SpeakerClustering with algorithm: {algorithm}")

    def cluster_embeddings(self, embeddings, timestamps=None):
        """
        Cluster speaker embeddings into speaker groups
        Args:
            embeddings: numpy array of embeddings (n_segments, embedding_dim)
        Returns:
            dict: Clustering results with labels and metrics
        """
        n = len(embeddings)
        if n == 0:
            logger.warning("No embeddings provided for clustering")
            return {'labels': [], 'n_clusters': 0, 'silhouette_score': 0.0, 'algorithm': self.algorithm}
        if n == 1:
            logger.info("Single embedding provided, assigning to cluster 0")
            return {'labels': [0], 'n_clusters': 1, 'silhouette_score': 1.0, 'algorithm': self.algorithm}
        logger.info(f"Clustering {n} embeddings using {self.algorithm}")
        try:
            if self.algorithm == 'agglomerative':
                return self._agglomerative_clustering(embeddings)
            elif self.algorithm == 'dbscan':
                return self._dbscan_clustering(embeddings)
            else:
                raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {'labels': [0] * n, 'n_clusters': 1, 'silhouette_score': 0.0, 'algorithm': self.algorithm, 'error': str(e)}

    def _agglomerative_clustering(self, embeddings):
        n_clusters = self._determine_optimal_clusters(embeddings)
        if self.affinity == 'cosine':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='cosine')
        else:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=self.linkage_method, metric=self.affinity)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = clusterer.fit_predict(embeddings)
        silhouette = self._calculate_silhouette_score(embeddings, labels)
        labels = self._relabel_clusters(labels)
        result = {'labels': labels.tolist(), 'n_clusters': len(np.unique(labels)), 'silhouette_score': silhouette, 'algorithm': 'agglomerative'}
        logger.info(f"Agglomerative clustering: {result['n_clusters']} clusters, silhouette: {result['silhouette_score']:.3f}")
        return result

    def _dbscan_clustering(self, embeddings):
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        clusterer = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='cosine')
        labels = clusterer.fit_predict(normalized)
        if np.any(labels == -1):
            labels = self._handle_noise_points(labels, normalized)
        silhouette = self._calculate_silhouette_score(normalized, labels)
        labels = self._relabel_clusters(labels)
        result = {'labels': labels.tolist(), 'n_clusters': len(np.unique(labels)), 'silhouette_score': silhouette, 'algorithm': 'dbscan'}
        logger.info(f"DBSCAN clustering: {result['n_clusters']} clusters, silhouette: {result['silhouette_score']:.3f}")
        return result

    def _determine_optimal_clusters(self, embeddings):
        n_samples = len(embeddings)
        min_c = max(1, self.min_speakers)
        max_c = min(n_samples, self.max_speakers)
        if min_c == max_c or not self.enable_adaptive_clustering:
            return min_c
        best_score, best_n = -1, min_c
        for n_clusters in range(min_c, max_c + 1):
            try:
                if self.affinity == 'cosine':
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', metric='cosine')
                else:
                    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=self.linkage_method, metric=self.affinity)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = clusterer.fit_predict(embeddings)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score, best_n = score, n_clusters
            except Exception as e:
                logger.debug(f"Failed to evaluate {n_clusters} clusters: {e}")
        logger.info(f"Optimal clusters: {best_n} (score: {best_score:.3f})")
        return best_n

    def _calculate_silhouette_score(self, embeddings, labels):
        try:
            if len(np.unique(labels)) <= 1:
                return 0.0
            return silhouette_score(embeddings, labels)
        except Exception as e:
            logger.debug(f"Failed to calculate silhouette score: {e}")
            return 0.0

    def _handle_noise_points(self, labels, embeddings):
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        unique_labels = np.unique(labels[~noise_mask])
        if len(unique_labels) == 0:
            return np.zeros_like(labels)
        centers = {label: np.mean(embeddings[labels == label], axis=0) for label in unique_labels}
        for i, is_noise in enumerate(noise_mask):
            if is_noise:
                nearest = min(centers, key=lambda l: np.linalg.norm(embeddings[i] - centers[l]))
                labels[i] = nearest
        return labels

    def _relabel_clusters(self, labels):
        unique = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique)}
        return np.array([label_map[label] for label in labels])

    def update_parameters(self, **kwargs):
        allowed = ['min_speakers', 'max_speakers', 'cluster_threshold', 'algorithm']
        updated = []
        for key in allowed:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                updated.append(key)
        if updated:
            logger.info(f"Updated clustering parameters: {updated}")

    def get_cluster_statistics(self, embeddings, labels):
        if len(embeddings) == 0 or len(labels) == 0:
            return {}
        unique_labels = np.unique(labels)
        stats = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': {},
            'cluster_compactness': {},
            'inter_cluster_distances': {}
        }
        for label in unique_labels:
            cluster_mask = np.array(labels) == label
            cluster_embeddings = embeddings[cluster_mask]
            stats['cluster_sizes'][int(label)] = len(cluster_embeddings)
            stats['cluster_compactness'][int(label)] = np.mean(pdist(cluster_embeddings, metric='cosine')) if len(cluster_embeddings) > 1 else 0.0
        if len(unique_labels) > 1:
            centers = {label: np.mean(embeddings[np.array(labels) == label], axis=0) for label in unique_labels}
            for i, l1 in enumerate(unique_labels):
                for l2 in unique_labels[i+1:]:
                    dist = np.linalg.norm(centers[l1] - centers[l2])
                    stats['inter_cluster_distances'][f'{int(l1)}-{int(l2)}'] = dist
        return stats 