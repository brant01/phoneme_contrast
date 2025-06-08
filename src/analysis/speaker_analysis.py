"""
Speaker invariance analysis for phoneme embeddings.

Tests whether the model learns phoneme representations that are invariant
to speaker characteristics (gender, individual variations).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class SpeakerInvarianceAnalyzer:
    """Analyze speaker invariance of phoneme embeddings."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the analyzer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def analyze_gender_invariance(
        self, embeddings: np.ndarray, phoneme_labels: np.ndarray, gender_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze how well embeddings are invariant to gender.

        Args:
            embeddings: Feature embeddings [n_samples, n_features]
            phoneme_labels: Phoneme class labels
            gender_labels: Gender labels (0=male, 1=female)

        Returns:
            Dictionary with invariance metrics
        """
        results = {}

        # 1. Gender classification accuracy (should be low for good invariance)
        gender_clf = SVC(kernel="linear", random_state=self.random_state)
        gender_scores = cross_val_score(gender_clf, embeddings, gender_labels, cv=5)
        results["gender_classification_accuracy"] = np.mean(gender_scores)
        results["gender_classification_std"] = np.std(gender_scores)

        # 2. Within-phoneme gender separation (should be low)
        unique_phonemes = np.unique(phoneme_labels)
        gender_separations = []

        for phoneme in unique_phonemes:
            phoneme_mask = phoneme_labels == phoneme
            phoneme_embeddings = embeddings[phoneme_mask]
            phoneme_genders = gender_labels[phoneme_mask]

            if len(np.unique(phoneme_genders)) == 2:  # Both genders present
                male_embeddings = phoneme_embeddings[phoneme_genders == 0]
                female_embeddings = phoneme_embeddings[phoneme_genders == 1]

                # Calculate average distance between genders
                if len(male_embeddings) > 0 and len(female_embeddings) > 0:
                    male_center = np.mean(male_embeddings, axis=0)
                    female_center = np.mean(female_embeddings, axis=0)
                    gender_distance = np.linalg.norm(male_center - female_center)
                    gender_separations.append(gender_distance)

        results["mean_within_phoneme_gender_distance"] = np.mean(gender_separations)
        results["std_within_phoneme_gender_distance"] = np.std(gender_separations)

        # 3. Compare phoneme classification accuracy by gender
        male_mask = gender_labels == 0
        female_mask = gender_labels == 1

        # Train on one gender, test on another
        if np.sum(male_mask) > 10 and np.sum(female_mask) > 10:
            # Male → Female
            phoneme_clf = SVC(kernel="linear", random_state=self.random_state)
            phoneme_clf.fit(embeddings[male_mask], phoneme_labels[male_mask])
            male_to_female_acc = phoneme_clf.score(embeddings[female_mask], phoneme_labels[female_mask])

            # Female → Male
            phoneme_clf = SVC(kernel="linear", random_state=self.random_state)
            phoneme_clf.fit(embeddings[female_mask], phoneme_labels[female_mask])
            female_to_male_acc = phoneme_clf.score(embeddings[male_mask], phoneme_labels[male_mask])

            results["cross_gender_accuracy_m2f"] = male_to_female_acc
            results["cross_gender_accuracy_f2m"] = female_to_male_acc
            results["cross_gender_accuracy_mean"] = (male_to_female_acc + female_to_male_acc) / 2

        # 4. Gender bias in clustering (silhouette score)
        if len(np.unique(gender_labels)) > 1:
            gender_silhouette = silhouette_score(embeddings, gender_labels)
            results["gender_clustering_silhouette"] = gender_silhouette

        # 5. Statistical test for gender effect on embeddings
        # Using MANOVA (Multivariate ANOVA) approximation
        male_embeddings = embeddings[male_mask]
        female_embeddings = embeddings[female_mask]

        if len(male_embeddings) > 1 and len(female_embeddings) > 1:
            # Hotelling's T-squared test (multivariate t-test)
            t2_stat, p_value = self._hotellings_t2_test(male_embeddings, female_embeddings)
            results["gender_effect_t2_statistic"] = t2_stat
            results["gender_effect_p_value"] = p_value
            results["gender_effect_significant"] = p_value < 0.05

        return results

    def analyze_speaker_clustering(
        self,
        embeddings: np.ndarray,
        phoneme_labels: np.ndarray,
        speaker_labels: np.ndarray,
        gender_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Analyze whether embeddings cluster by speaker rather than phoneme.

        Args:
            embeddings: Feature embeddings
            phoneme_labels: Phoneme class labels
            speaker_labels: Individual speaker IDs
            gender_labels: Optional gender labels

        Returns:
            Dictionary with clustering metrics
        """
        results = {}

        # 1. Silhouette scores for different groupings
        # Higher score = better clustering
        phoneme_silhouette = silhouette_score(embeddings, phoneme_labels)
        speaker_silhouette = silhouette_score(embeddings, speaker_labels)

        results["phoneme_clustering_score"] = phoneme_silhouette
        results["speaker_clustering_score"] = speaker_silhouette
        results["clustering_ratio"] = phoneme_silhouette / (speaker_silhouette + 1e-10)

        # 2. Speaker classification accuracy (should be low)
        if len(np.unique(speaker_labels)) > 2:
            speaker_clf = SVC(kernel="linear", random_state=self.random_state)
            speaker_scores = cross_val_score(speaker_clf, embeddings, speaker_labels, cv=3)
            results["speaker_classification_accuracy"] = np.mean(speaker_scores)

        # 3. Within-phoneme speaker variance
        unique_phonemes = np.unique(phoneme_labels)
        within_phoneme_variances = []
        between_phoneme_variances = []

        for phoneme in unique_phonemes:
            phoneme_mask = phoneme_labels == phoneme
            phoneme_embeddings = embeddings[phoneme_mask]

            if len(phoneme_embeddings) > 1:
                # Variance within this phoneme class
                within_var = np.mean(np.var(phoneme_embeddings, axis=0))
                within_phoneme_variances.append(within_var)

                # Distance to other phoneme centers
                other_mask = phoneme_labels != phoneme
                if np.any(other_mask):
                    other_center = np.mean(embeddings[other_mask], axis=0)
                    phoneme_center = np.mean(phoneme_embeddings, axis=0)
                    between_var = np.linalg.norm(phoneme_center - other_center) ** 2
                    between_phoneme_variances.append(between_var)

        results["mean_within_phoneme_variance"] = np.mean(within_phoneme_variances)
        results["mean_between_phoneme_variance"] = np.mean(between_phoneme_variances)
        results["variance_ratio"] = (
            np.mean(between_phoneme_variances) / (np.mean(within_phoneme_variances) + 1e-10)
        )

        return results

    def analyze_phoneme_consistency_across_speakers(
        self,
        embeddings: np.ndarray,
        phoneme_labels: np.ndarray,
        speaker_labels: np.ndarray,
        n_speakers_test: int = 2,
    ) -> Dict[str, float]:
        """
        Test if phoneme representations are consistent across speakers.

        Args:
            embeddings: Feature embeddings
            phoneme_labels: Phoneme class labels
            speaker_labels: Speaker IDs
            n_speakers_test: Number of speakers to hold out for testing

        Returns:
            Dictionary with consistency metrics
        """
        unique_speakers = np.unique(speaker_labels)
        if len(unique_speakers) < n_speakers_test + 2:
            self.logger.warning("Not enough speakers for leave-speakers-out analysis")
            return {}

        # Leave-speakers-out cross-validation
        accuracies = []
        n_iterations = min(10, len(unique_speakers) // n_speakers_test)

        rng = np.random.RandomState(self.random_state)
        for i in range(n_iterations):
            # Randomly select test speakers
            test_speakers = rng.choice(unique_speakers, n_speakers_test, replace=False)
            test_mask = np.isin(speaker_labels, test_speakers)
            train_mask = ~test_mask

            if np.sum(train_mask) > 10 and np.sum(test_mask) > 5:
                # Train classifier on remaining speakers
                clf = SVC(kernel="linear", random_state=self.random_state)
                clf.fit(embeddings[train_mask], phoneme_labels[train_mask])

                # Test on held-out speakers
                acc = clf.score(embeddings[test_mask], phoneme_labels[test_mask])
                accuracies.append(acc)

        results = {
            "leave_speakers_out_accuracy": np.mean(accuracies),
            "leave_speakers_out_std": np.std(accuracies),
            "n_test_iterations": len(accuracies),
        }

        return results

    def compute_speaker_adaptation_score(
        self, embeddings: np.ndarray, phoneme_labels: np.ndarray, speaker_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute how much adaptation is needed for new speakers.

        Uses LDA to measure separability with/without speaker information.

        Args:
            embeddings: Feature embeddings
            phoneme_labels: Phoneme class labels
            speaker_labels: Speaker IDs

        Returns:
            Dictionary with adaptation metrics
        """
        # 1. LDA with phoneme labels only
        lda_phoneme = LinearDiscriminantAnalysis()
        phoneme_scores = cross_val_score(lda_phoneme, embeddings, phoneme_labels, cv=5)

        # 2. Create combined labels (phoneme + speaker)
        combined_labels = [f"{p}_{s}" for p, s in zip(phoneme_labels, speaker_labels)]
        unique_combined = np.unique(combined_labels)

        # Only proceed if we have reasonable number of combined classes
        if len(unique_combined) < 100:  # Arbitrary threshold
            lda_combined = LinearDiscriminantAnalysis()
            combined_scores = cross_val_score(lda_combined, embeddings, combined_labels, cv=3)

            adaptation_benefit = np.mean(combined_scores) - np.mean(phoneme_scores)
        else:
            adaptation_benefit = None

        results = {
            "phoneme_only_lda_accuracy": np.mean(phoneme_scores),
            "phoneme_only_lda_std": np.std(phoneme_scores),
        }

        if adaptation_benefit is not None:
            results["speaker_specific_lda_accuracy"] = np.mean(combined_scores)
            results["adaptation_benefit"] = adaptation_benefit

        return results

    def _hotellings_t2_test(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Hotelling's T-squared test (multivariate t-test).

        Args:
            group1: First group [n1, p]
            group2: Second group [n2, p]

        Returns:
            (t2_statistic, p_value)
        """
        n1, p = group1.shape
        n2, _ = group2.shape

        # Calculate means
        mean1 = np.mean(group1, axis=0)
        mean2 = np.mean(group2, axis=0)
        mean_diff = mean1 - mean2

        # Calculate pooled covariance
        cov1 = np.cov(group1.T)
        cov2 = np.cov(group2.T)
        pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

        # Add small regularization for numerical stability
        pooled_cov += np.eye(p) * 1e-6

        # Calculate T-squared statistic
        inv_cov = np.linalg.inv(pooled_cov)
        t2 = (n1 * n2) / (n1 + n2) * mean_diff @ inv_cov @ mean_diff

        # Convert to F-statistic
        f_stat = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * t2

        # Calculate p-value
        df1 = p
        df2 = n1 + n2 - p - 1
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        return t2, p_value