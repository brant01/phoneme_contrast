"""
Phonetic feature analysis for understanding learned representations.

Analyzes how embeddings organize according to phonetic features like:
- Place of articulation (labial, dental, velar, etc.)
- Manner of articulation (stop, fricative, etc.)
- Voicing (voiced vs unvoiced)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans


class PhoneticFeatureAnalyzer:
    """Analyze phoneme embeddings based on phonetic features."""

    # Phonetic feature definitions for the dataset
    PHONETIC_FEATURES = {
        # Place of articulation
        "labial": ["pa", "pe", "pi", "pu", "ba", "bi", "bu", "fa", "fe", "fi", "fu", "apa"],
        "dental": ["ta", "te", "ti", "tu", "da", "de", "di", "du", "sa", "se", "si", "su"],
        "velar": ["ka", "ke", "ki", "ku", "ga", "ge", "gi", "gu"],
        
        # Manner of articulation
        "stop": ["pa", "pe", "pi", "pu", "ba", "bi", "bu", "ta", "te", "ti", "tu", 
                 "da", "de", "di", "du", "ka", "ke", "ki", "ku", "ga", "ge", "gi", "gu"],
        "fricative": ["fa", "fe", "fi", "fu", "sa", "se", "si", "su"],
        
        # Voicing
        "voiced": ["ba", "bi", "bu", "da", "de", "di", "du", "ga", "ge", "gi", "gu"],
        "unvoiced": ["pa", "pe", "pi", "pu", "ta", "te", "ti", "tu", "ka", "ke", "ki", "ku",
                     "fa", "fe", "fi", "fu", "sa", "se", "si", "su"],
        
        # Vowel context
        "vowel_a": ["pa", "ba", "ta", "da", "ka", "ga", "fa", "sa", "ada", "apa"],
        "vowel_e": ["pe", "de", "te", "ke", "ge", "fe", "se", "ege", "ete"],
        "vowel_i": ["pi", "bi", "di", "ti", "ki", "gi", "fi", "si", "ibi", "isi"],
        "vowel_u": ["pu", "bu", "du", "tu", "ku", "gu", "fu", "su", "ubu", "uku"],
        
        # Structure
        "cv": ["pa", "pe", "pi", "pu", "ba", "bi", "bu", "ta", "te", "ti", "tu",
               "da", "de", "di", "du", "ka", "ke", "ki", "ku", "ga", "ge", "gi", "gu",
               "fa", "fe", "fi", "fu", "sa", "se", "si", "su"],
        "vcv": ["ada", "apa", "ege", "ete", "ibi", "isi", "ubu", "uku"],
    }

    def __init__(self, random_state: int = 42):
        """
        Initialize the analyzer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def analyze_feature_clustering(
        self,
        embeddings: np.ndarray,
        phoneme_labels: List[str],
        features_to_analyze: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze how well embeddings cluster according to phonetic features.

        Args:
            embeddings: Phoneme embeddings [n_samples, n_features]
            phoneme_labels: List of phoneme strings (e.g., ["pa", "ba", ...])
            features_to_analyze: List of features to analyze (default: all)

        Returns:
            Dictionary mapping feature names to clustering metrics
        """
        if features_to_analyze is None:
            features_to_analyze = list(self.PHONETIC_FEATURES.keys())

        results = {}
        
        for feature_name in features_to_analyze:
            if feature_name not in self.PHONETIC_FEATURES:
                self.logger.warning(f"Unknown feature: {feature_name}")
                continue

            # Get phonemes with this feature
            feature_phonemes = set(self.PHONETIC_FEATURES[feature_name])
            
            # Create binary labels (has feature vs doesn't have feature)
            feature_labels = np.array([
                1 if phoneme in feature_phonemes else 0
                for phoneme in phoneme_labels
            ])

            # Skip if all phonemes have or don't have the feature
            if len(np.unique(feature_labels)) < 2:
                continue

            # Calculate metrics
            metrics = self._calculate_feature_metrics(embeddings, feature_labels, feature_name)
            results[feature_name] = metrics

        return results

    def analyze_feature_distances(
        self,
        embeddings: np.ndarray,
        phoneme_labels: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Analyze distances between phonemes sharing/not sharing features.

        Args:
            embeddings: Phoneme embeddings
            phoneme_labels: List of phoneme strings

        Returns:
            Dictionary with distance analysis results
        """
        # Compute pairwise distances
        distances = squareform(pdist(embeddings))
        n_samples = len(phoneme_labels)

        results = {}

        for feature_name, feature_phonemes in self.PHONETIC_FEATURES.items():
            feature_phonemes = set(feature_phonemes)
            
            # Create masks for phonemes with/without feature
            has_feature = np.array([p in feature_phonemes for p in phoneme_labels])
            
            if np.sum(has_feature) < 2 or np.sum(~has_feature) < 2:
                continue

            # Collect distances
            within_feature_dists = []
            across_feature_dists = []
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if has_feature[i] == has_feature[j]:
                        within_feature_dists.append(distances[i, j])
                    else:
                        across_feature_dists.append(distances[i, j])

            results[feature_name] = {
                "within_feature_distances": np.array(within_feature_dists),
                "across_feature_distances": np.array(across_feature_dists),
                "mean_within": np.mean(within_feature_dists),
                "mean_across": np.mean(across_feature_dists),
                "distance_ratio": np.mean(across_feature_dists) / (np.mean(within_feature_dists) + 1e-10),
            }

        return results

    def analyze_feature_hierarchy(
        self,
        embeddings: np.ndarray,
        phoneme_labels: List[str],
    ) -> Dict[str, float]:
        """
        Analyze hierarchical organization of features.

        Tests if broad features (e.g., manner) are more salient than narrow ones (e.g., place).

        Args:
            embeddings: Phoneme embeddings
            phoneme_labels: List of phoneme strings

        Returns:
            Dictionary with hierarchy analysis
        """
        # Define feature hierarchy (broad to narrow)
        hierarchy = [
            ("structure", ["cv", "vcv"]),
            ("manner", ["stop", "fricative"]),
            ("voicing", ["voiced", "unvoiced"]),
            ("place", ["labial", "dental", "velar"]),
            ("vowel", ["vowel_a", "vowel_e", "vowel_i", "vowel_u"]),
        ]

        results = {}
        distance_ratios = []

        for level_name, features in hierarchy:
            level_ratios = []
            
            for feature in features:
                if feature in self.PHONETIC_FEATURES:
                    feature_phonemes = set(self.PHONETIC_FEATURES[feature])
                    feature_labels = np.array([
                        1 if p in feature_phonemes else 0 for p in phoneme_labels
                    ])
                    
                    if len(np.unique(feature_labels)) == 2:
                        metrics = self._calculate_feature_metrics(
                            embeddings, feature_labels, feature
                        )
                        if "distance_ratio" in metrics:
                            level_ratios.append(metrics["distance_ratio"])

            if level_ratios:
                results[f"{level_name}_mean_ratio"] = np.mean(level_ratios)
                distance_ratios.append((level_name, np.mean(level_ratios)))

        # Calculate hierarchy correlation
        if len(distance_ratios) > 2:
            # Test if distance ratios decrease with hierarchy level (broader = higher ratio)
            levels = list(range(len(distance_ratios)))
            ratios = [r[1] for r in distance_ratios]
            correlation, p_value = spearmanr(levels, ratios)
            
            results["hierarchy_correlation"] = correlation
            results["hierarchy_p_value"] = p_value

        return results

    def find_phoneme_clusters(
        self,
        embeddings: np.ndarray,
        phoneme_labels: List[str],
        n_clusters: int = 10,
    ) -> Dict[str, any]:
        """
        Find natural clusters in embedding space and analyze their phonetic composition.

        Args:
            embeddings: Phoneme embeddings
            phoneme_labels: List of phoneme strings
            n_clusters: Number of clusters to find

        Returns:
            Dictionary with clustering results
        """
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Analyze cluster composition
        clusters = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_phonemes = [p for p, m in zip(phoneme_labels, cluster_mask) if m]
            
            # Analyze features in this cluster
            cluster_features = self._analyze_cluster_features(cluster_phonemes)
            
            clusters[f"cluster_{i}"] = {
                "phonemes": cluster_phonemes,
                "size": len(cluster_phonemes),
                "dominant_features": cluster_features,
                "center": kmeans.cluster_centers_[i],
            }

        # Calculate clustering quality metrics
        # Compare with feature-based groupings
        feature_scores = {}
        for feature_name, feature_phonemes in self.PHONETIC_FEATURES.items():
            feature_labels = np.array([
                1 if p in feature_phonemes else 0 for p in phoneme_labels
            ])
            
            if len(np.unique(feature_labels)) >= 2:
                ari = adjusted_rand_score(feature_labels, cluster_labels)
                nmi = normalized_mutual_info_score(feature_labels, cluster_labels)
                feature_scores[feature_name] = {"ari": ari, "nmi": nmi}

        return {
            "clusters": clusters,
            "feature_alignment_scores": feature_scores,
            "n_clusters": n_clusters,
        }

    def _calculate_feature_metrics(
        self,
        embeddings: np.ndarray,
        feature_labels: np.ndarray,
        feature_name: str,
    ) -> Dict[str, float]:
        """Calculate clustering metrics for a phonetic feature."""
        # Get embeddings for each group
        positive_embeddings = embeddings[feature_labels == 1]
        negative_embeddings = embeddings[feature_labels == 0]

        metrics = {}

        # 1. Mean distance ratio
        if len(positive_embeddings) > 1 and len(negative_embeddings) > 1:
            # Within-group distances
            pos_distances = pdist(positive_embeddings)
            neg_distances = pdist(negative_embeddings)
            within_distances = np.concatenate([pos_distances, neg_distances])

            # Between-group distances
            between_distances = []
            for pos_emb in positive_embeddings:
                for neg_emb in negative_embeddings:
                    between_distances.append(np.linalg.norm(pos_emb - neg_emb))

            metrics["mean_within_distance"] = np.mean(within_distances)
            metrics["mean_between_distance"] = np.mean(between_distances)
            metrics["distance_ratio"] = np.mean(between_distances) / (np.mean(within_distances) + 1e-10)

        # 2. Centroid distance
        pos_center = np.mean(positive_embeddings, axis=0)
        neg_center = np.mean(negative_embeddings, axis=0)
        metrics["centroid_distance"] = np.linalg.norm(pos_center - neg_center)

        # 3. Linear separability (using simple linear classifier)
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        
        clf = SVC(kernel="linear", random_state=self.random_state)
        scores = cross_val_score(clf, embeddings, feature_labels, cv=3)
        metrics["linear_separability"] = np.mean(scores)

        return metrics

    def _analyze_cluster_features(self, cluster_phonemes: List[str]) -> Dict[str, float]:
        """Analyze which phonetic features are dominant in a cluster."""
        cluster_phonemes_set = set(cluster_phonemes)
        feature_scores = {}

        for feature_name, feature_phonemes in self.PHONETIC_FEATURES.items():
            feature_phonemes_set = set(feature_phonemes)
            
            # Calculate overlap
            overlap = len(cluster_phonemes_set & feature_phonemes_set)
            if overlap > 0:
                # Precision: what fraction of cluster has this feature
                precision = overlap / len(cluster_phonemes_set)
                # Recall: what fraction of feature phonemes are in cluster
                recall = overlap / len(feature_phonemes_set)
                # F1 score
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                
                feature_scores[feature_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

        # Sort by F1 score
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1]["f1"],
            reverse=True
        )

        return dict(sorted_features[:3])  # Return top 3 features

    def generate_feature_report(
        self,
        embeddings: np.ndarray,
        phoneme_labels: List[str],
    ) -> str:
        """
        Generate a comprehensive report on phonetic feature organization.

        Args:
            embeddings: Phoneme embeddings
            phoneme_labels: List of phoneme strings

        Returns:
            Formatted report string
        """
        report = "## Phonetic Feature Analysis Report\n\n"

        # 1. Feature clustering analysis
        report += "### Feature Clustering Analysis\n"
        clustering_results = self.analyze_feature_clustering(embeddings, phoneme_labels)
        
        for feature, metrics in clustering_results.items():
            report += f"\n**{feature}**:\n"
            report += f"- Linear separability: {metrics.get('linear_separability', 0):.3f}\n"
            report += f"- Distance ratio: {metrics.get('distance_ratio', 0):.3f}\n"
            report += f"- Centroid distance: {metrics.get('centroid_distance', 0):.3f}\n"

        # 2. Feature hierarchy
        report += "\n### Feature Hierarchy Analysis\n"
        hierarchy_results = self.analyze_feature_hierarchy(embeddings, phoneme_labels)
        
        for key, value in hierarchy_results.items():
            if "ratio" in key:
                report += f"- {key}: {value:.3f}\n"

        # 3. Natural clusters
        report += "\n### Natural Clustering Analysis\n"
        cluster_results = self.find_phoneme_clusters(embeddings, phoneme_labels, n_clusters=8)
        
        report += f"Found {cluster_results['n_clusters']} clusters\n\n"
        
        # Best feature alignments
        feature_scores = cluster_results["feature_alignment_scores"]
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1]["nmi"],
            reverse=True
        )
        
        report += "Best feature alignments (by NMI):\n"
        for feature, scores in sorted_features[:5]:
            report += f"- {feature}: NMI={scores['nmi']:.3f}, ARI={scores['ari']:.3f}\n"

        return report