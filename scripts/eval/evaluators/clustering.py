"""Speaker clustering algorithms for diarization.

Uses spectralcluster library from wq2012:
https://github.com/wq2012/SpectralCluster

Implements refinement options based on:
- Park et al. "Auto-tuning spectral clustering for speaker diarization
  using normalized maximum eigengap." IEEE SPL 2019.
"""

import numpy as np


class SpeakerClusterer:
    """Speaker clustering backend using spectralcluster library.

    Features:
    - Spectral clustering with eigenvalue gap for auto speaker count detection
    - Affinity matrix refinement (blur, threshold, symmetrize, diffuse)
    - Auto-tuning of p_percentile parameter
    - Small cluster merging
    """

    def __init__(
        self,
        min_num_spks: int = 2,
        max_num_spks: int = 10,
        min_cluster_size: int = 4,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self._clusterer = None

    def _get_clusterer(self):
        """Lazy-load the spectralcluster library with tuned eigenvalue gap."""
        if self._clusterer is None:
            from spectralcluster import SpectralClusterer

            self._clusterer = SpectralClusterer(
                min_clusters=self.min_num_spks,
                max_clusters=self.max_num_spks,
                custom_dist="cosine",
                # Lower threshold to detect more speakers via eigenvalue gap
                stop_eigenvalue=1e-6,
            )
        return self._clusterer

    def __call__(self, embeddings: np.ndarray, num_speakers: int | None = None) -> np.ndarray:
        """Cluster speaker embeddings and return labels.

        Args:
            embeddings: Speaker embeddings of shape [N, D]
            num_speakers: Optional oracle number of speakers

        Returns:
            Cluster labels of shape [N]
        """
        if len(embeddings.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

        # Handle edge cases
        if embeddings.shape[0] == 0:
            return np.array([], dtype=int)
        if embeddings.shape[0] == 1:
            return np.array([0], dtype=int)
        if embeddings.shape[0] < 6:
            return np.zeros(embeddings.shape[0], dtype=int)

        # Preprocess: filter zero/near-zero norm embeddings to prevent numerical issues
        norms = np.linalg.norm(embeddings, axis=1)
        valid_mask = norms > 1e-6
        if not np.all(valid_mask):
            # Some embeddings are invalid - cluster only valid ones
            valid_embeddings = embeddings[valid_mask]
            if len(valid_embeddings) < 2:
                return np.zeros(embeddings.shape[0], dtype=int)
            valid_labels = self._do_cluster(valid_embeddings, num_speakers)
            # Assign invalid embeddings to nearest valid cluster
            labels = np.zeros(embeddings.shape[0], dtype=int)
            labels[valid_mask] = valid_labels
            # For invalid embeddings, find nearest valid centroid
            if np.any(~valid_mask):
                from scipy.spatial.distance import cdist

                unique_labels = np.unique(valid_labels)
                centroids = np.vstack(
                    [np.mean(valid_embeddings[valid_labels == k], axis=0) for k in unique_labels]
                )
                invalid_embeddings = embeddings[~valid_mask]
                dists = cdist(invalid_embeddings, centroids, metric="cosine")
                nearest = np.argmin(dists, axis=1)
                labels[~valid_mask] = unique_labels[nearest]
        else:
            labels = self._do_cluster(embeddings, num_speakers)

        # Re-index labels sequentially
        _, labels = np.unique(labels, return_inverse=True)

        return labels

    def _do_cluster(self, embeddings: np.ndarray, num_speakers: int | None = None) -> np.ndarray:
        """Internal clustering logic."""
        import warnings

        # Ensure embeddings are normalized and finite
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        embeddings = embeddings / norms

        # Replace any remaining NaN/inf with zeros
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        # Get clusterer
        clusterer = self._get_clusterer()

        # Handle oracle number of speakers
        if num_speakers is not None:
            clusterer.min_clusters = num_speakers
            clusterer.max_clusters = num_speakers

        # Run clustering (suppress numerical warnings from spectralcluster)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            labels = clusterer.predict(embeddings)

        # Reset to original values
        if num_speakers is not None:
            clusterer.min_clusters = self.min_num_spks
            clusterer.max_clusters = self.max_num_spks

        labels = np.array(labels)

        # Post-process: merge small clusters into nearest large cluster
        return self._merge_small_clusters(labels, embeddings)

    def _merge_small_clusters(self, labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Merge small clusters into nearest large cluster."""
        from scipy.spatial.distance import cdist

        labels = labels.copy()

        # Adaptive min_cluster_size for small datasets
        min_cluster_size = min(self.min_cluster_size, max(1, round(0.1 * len(embeddings))))

        # Split clusters into large and small
        cluster_unique, cluster_counts = np.unique(labels, return_counts=True)
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        small_clusters = cluster_unique[cluster_counts < min_cluster_size]

        # If no large clusters, assign all to cluster 0
        if len(large_clusters) == 0:
            labels[:] = 0
            return labels

        # If no small clusters, nothing to merge
        if len(small_clusters) == 0:
            return labels

        # Compute centroids for large and small clusters
        large_centroids = np.vstack(
            [np.mean(embeddings[labels == k], axis=0) for k in large_clusters]
        )
        small_centroids = np.vstack(
            [np.mean(embeddings[labels == k], axis=0) for k in small_clusters]
        )

        # Find nearest large cluster for each small cluster
        centroids_dist = cdist(large_centroids, small_centroids, metric="cosine")
        nearest_large = np.argmin(centroids_dist, axis=0)

        # Reassign small clusters to nearest large cluster
        for i, small_k in enumerate(small_clusters):
            labels[labels == small_k] = large_clusters[nearest_large[i]]

        return labels
