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
    def __init__(self, algorithm='autoscan'):
        self.algorithm = algorithm
        
        # Static configuration that doesn't change during runtime
        self.cluster_threshold = CONFIG.get('cluster_threshold', 0.3)
        self.linkage_method = CONFIG.get('clustering_linkage', 'ward')
        self.affinity = CONFIG.get('clustering_affinity', 'euclidean')
        
        # Autoscan parameters
        self.autoscan_k_neighbors = CONFIG.get('autoscan_k_neighbors', 4)
        self.autoscan_min_eps = CONFIG.get('autoscan_min_eps', 0.1)
        self.autoscan_max_eps = CONFIG.get('autoscan_max_eps', 0.9)
        self.autoscan_elbow_method = CONFIG.get('autoscan_elbow_method', 'knee')
        self.autoscan_enable_speaker_constraints = CONFIG.get('autoscan_enable_speaker_constraints', True)
        
        self.enable_adaptive_clustering = CONFIG.get('enable_adaptive_clustering', True)
        logger.info(f"Initialized SpeakerClustering with algorithm: {algorithm}")

    def _get_min_speakers(self):
        """Get current minimum speakers setting dynamically"""
        return CONFIG.get('min_speakers', 1)
    
    def _get_max_speakers(self):
        """Get current maximum speakers setting dynamically"""
        return CONFIG.get('max_speakers', 10)

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
            elif self.algorithm == 'autoscan':
                return self._autoscan_clustering(embeddings)
            else:
                raise ValueError(f"Unknown clustering algorithm: {self.algorithm}. Supported: 'agglomerative', 'autoscan'")
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

    def _autoscan_clustering(self, embeddings):
        """DBSCAN clustering with automatic parameter tuning"""
        eps, min_samples = self._autoscan_parameters(embeddings)
        
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clusterer.fit_predict(normalized)
        
        # Enhanced noise handling with speaker constraints
        labels = self._handle_noise_points_autoscan(labels, normalized)
        
        silhouette = self._calculate_silhouette_score(normalized, labels)
        labels = self._relabel_clusters(labels)
        
        result = {
            'labels': labels.tolist(), 
            'n_clusters': len(np.unique(labels)), 
            'silhouette_score': silhouette, 
            'algorithm': 'autoscan',
            'eps_used': eps,
            'min_samples_used': min_samples
        }
        
        logger.info(f"Autoscan clustering: {result['n_clusters']} clusters, eps={eps:.3f}, min_samples={min_samples}")
        return result

    def _determine_optimal_clusters(self, embeddings):
        n_samples = len(embeddings)
        min_c = max(1, self._get_min_speakers())
        max_c = min(n_samples, self._get_max_speakers())
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



    def _relabel_clusters(self, labels):
        unique = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique)}
        return np.array([label_map[label] for label in labels])

    def _autoscan_parameters(self, embeddings):
        """Automatically determine optimal DBSCAN parameters"""
        from sklearn.neighbors import NearestNeighbors
        
        # Normalize embeddings for cosine distance
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate k-nearest neighbors distances
        k = min(self.autoscan_k_neighbors, len(embeddings) - 1)
        if k < 2:
            return self.autoscan_min_eps, 2
        
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine').fit(normalized)
        distances, _ = nbrs.kneighbors(normalized)
        
        # Find optimal eps using elbow method on sorted distances
        sorted_distances = np.sort(distances[:, -1])  # k-th nearest neighbor distances
        eps = self._find_elbow_point(sorted_distances)
        
        # Determine min_samples based on data size and expected speakers
        min_samples = max(2, min(len(embeddings) // 10, 5))
        
        # Apply speaker constraints if enabled
        if self.autoscan_enable_speaker_constraints:
            eps, min_samples = self._apply_speaker_constraints(eps, min_samples, len(embeddings))
        
        return eps, min_samples

    def _find_elbow_point(self, sorted_distances):
        """Find the elbow point in sorted distances using knee detection"""
        if len(sorted_distances) < 3:
            return self.autoscan_min_eps
        
        # Use knee detection method
        if self.autoscan_elbow_method == 'knee':
            eps = self._knee_detection(sorted_distances)
        else:
            # Fallback to percentile-based method
            eps = np.percentile(sorted_distances, 75)
        
        # Constrain eps to configured range
        eps = max(self.autoscan_min_eps, min(eps, self.autoscan_max_eps))
        
        return eps

    def _knee_detection(self, sorted_distances):
        """Detect knee point using curvature method"""
        if len(sorted_distances) < 3:
            return sorted_distances[-1] if len(sorted_distances) > 0 else self.autoscan_min_eps
        
        # Calculate curvature
        x = np.arange(len(sorted_distances))
        y = sorted_distances
        
        # First and second derivatives
        dy = np.gradient(y)
        d2y = np.gradient(dy)
        
        # Curvature
        curvature = np.abs(d2y) / (1 + dy**2)**1.5
        
        # Find point of maximum curvature
        knee_idx = np.argmax(curvature)
        knee_value = sorted_distances[knee_idx]
        
        return knee_value

    def _apply_speaker_constraints(self, eps, min_samples, n_samples):
        """Apply speaker count constraints to autoscan parameters"""
        # Adjust eps based on expected speaker count
        if self._get_min_speakers() == self._get_max_speakers():
            # Fixed speaker count - adjust eps to encourage that number of clusters
            target_clusters = self._get_min_speakers()
            if target_clusters == 1:
                # Single speaker - use tighter eps
                eps = min(eps, 0.3)
            elif target_clusters <= 3:
                # Small group - moderate eps
                eps = min(eps, 0.5)
            else:
                # Large group - more permissive eps
                eps = min(eps, 0.7)
        
        # Adjust min_samples based on data size and speaker expectations
        if n_samples < 10:
            min_samples = 2
        elif self._get_max_speakers() <= 2:
            min_samples = max(2, min_samples // 2)
        elif self._get_max_speakers() >= 5:
            min_samples = min(min_samples + 1, n_samples // 5)
        
        return eps, min_samples

    def _handle_noise_points_autoscan(self, labels, embeddings):
        """Enhanced noise point handling for autoscan with speaker constraints"""
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        unique_labels = np.unique(labels[~noise_mask])
        if len(unique_labels) == 0:
            return np.zeros_like(labels)
        
        # Calculate cluster centers
        centers = {label: np.mean(embeddings[labels == label], axis=0) for label in unique_labels}
        
        # Handle noise points with speaker constraints
        for i, is_noise in enumerate(noise_mask):
            if is_noise:
                # Find nearest cluster center
                nearest = min(centers, key=lambda l: np.linalg.norm(embeddings[i] - centers[l]))
                labels[i] = nearest
        
        # Apply speaker count constraints if enabled
        if self.autoscan_enable_speaker_constraints:
            labels = self._enforce_speaker_constraints(labels, embeddings)
        
        return labels

    def _enforce_speaker_constraints(self, labels, embeddings):
        """Enforce min/max speaker constraints by merging or splitting clusters"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters == 0:
            return labels
        
        # If we have too many clusters, merge the smallest ones
        if n_clusters > self._get_max_speakers():
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1])
            
            # Merge smallest clusters into largest ones
            for i in range(n_clusters - self._get_max_speakers()):
                smallest_cluster = sorted_clusters[i][0]
                largest_cluster = sorted_clusters[-1][0]
                labels[labels == smallest_cluster] = largest_cluster
        
        # If we have too few clusters and enough data, try to split
        elif n_clusters < self._get_min_speakers() and len(embeddings) >= self._get_min_speakers() * 3:
            # This is a simplified approach - in practice, you might want more sophisticated splitting
            largest_cluster = max(unique_labels, key=lambda l: np.sum(labels == l))
            largest_mask = labels == largest_cluster
            
            if np.sum(largest_mask) >= 6:  # Only split if we have enough points
                # Simple splitting: assign half to a new cluster
                largest_indices = np.where(largest_mask)[0]
                split_point = len(largest_indices) // 2
                
                new_cluster_id = max(unique_labels) + 1
                labels[largest_indices[split_point:]] = new_cluster_id
        
        return labels

    def update_parameters(self, **kwargs):
        allowed = ['min_speakers', 'max_speakers', 'cluster_threshold', 'algorithm']
        updated = []
        for key in allowed:
            if key in kwargs:
                setattr(self, key, kwargs[key])
                updated.append(key)
        
        # Update autoscan parameters if provided
        autoscan_params = ['autoscan_k_neighbors', 'autoscan_min_eps', 'autoscan_max_eps', 
                          'autoscan_elbow_method', 'autoscan_enable_speaker_constraints']
        for key in autoscan_params:
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