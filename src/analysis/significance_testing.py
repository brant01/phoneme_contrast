"""
Statistical significance testing for phoneme classification experiments.

Provides rigorous statistical analysis including:
- Bootstrap confidence intervals
- Permutation tests
- McNemar's test for classifier comparison
- Effect size calculations (Cohen's d)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils import resample


class SignificanceTester:
    """Perform statistical significance tests for model evaluation."""

    def __init__(self, n_bootstrap: int = 1000, n_permutations: int = 1000, random_state: int = 42):
        """
        Initialize the significance tester.

        Args:
            n_bootstrap: Number of bootstrap iterations
            n_permutations: Number of permutation test iterations
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.logger = logging.getLogger(__name__)

    def bootstrap_confidence_interval(
        self, y_true: np.ndarray, y_pred: np.ndarray, metric_fn=accuracy_score, alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_fn: Metric function (default: accuracy_score)
            alpha: Significance level (default: 0.05 for 95% CI)

        Returns:
            (metric_value, lower_bound, upper_bound)
        """
        n_samples = len(y_true)
        scores = []

        # Bootstrap resampling
        for _ in range(self.n_bootstrap):
            indices = self.rng.choice(n_samples, n_samples, replace=True)
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(score)

        # Calculate confidence interval
        metric_value = metric_fn(y_true, y_pred)
        lower = np.percentile(scores, 100 * alpha / 2)
        upper = np.percentile(scores, 100 * (1 - alpha / 2))

        return metric_value, lower, upper

    def permutation_test(
        self, X: np.ndarray, y: np.ndarray, model1, model2, cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform permutation test to compare two models.

        Args:
            X: Feature matrix
            y: Labels
            model1: First model
            model2: Second model
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with test results
        """
        # Get predictions using cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        pred1 = cross_val_predict(model1, X, y, cv=cv)
        pred2 = cross_val_predict(model2, X, y, cv=cv)

        # Calculate observed difference
        acc1 = accuracy_score(y, pred1)
        acc2 = accuracy_score(y, pred2)
        observed_diff = acc1 - acc2

        # Permutation test
        permuted_diffs = []
        for _ in range(self.n_permutations):
            # Randomly swap predictions between models
            swap_mask = self.rng.randint(0, 2, size=len(y)).astype(bool)
            perm_pred1 = np.where(swap_mask, pred2, pred1)
            perm_pred2 = np.where(swap_mask, pred1, pred2)

            perm_acc1 = accuracy_score(y, perm_pred1)
            perm_acc2 = accuracy_score(y, perm_pred2)
            permuted_diffs.append(perm_acc1 - perm_acc2)

        # Calculate p-value
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

        return {
            "model1_accuracy": acc1,
            "model2_accuracy": acc2,
            "observed_difference": observed_diff,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    def mcnemar_test(self, y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Dict[str, float]:
        """
        Perform McNemar's test to compare two classifiers.

        Args:
            y_true: True labels
            pred1: Predictions from first classifier
            pred2: Predictions from second classifier

        Returns:
            Dictionary with test results
        """
        # Create contingency table
        correct1 = pred1 == y_true
        correct2 = pred2 == y_true

        # Count discordant pairs
        n01 = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
        n10 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct

        # McNemar's test statistic
        if n01 + n10 == 0:
            # No discordant pairs
            return {
                "n01": n01,
                "n10": n10,
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
            }

        # Use continuity correction for small samples
        if n01 + n10 < 25:
            statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        else:
            statistic = (n01 - n10) ** 2 / (n01 + n10)

        # Calculate p-value (chi-square distribution with 1 df)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return {
            "n01": n01,
            "n10": n10,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group of measurements
            group2: Second group of measurements

        Returns:
            Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        d = (mean1 - mean2) / pooled_std

        return d

    def analyze_cross_validation_stability(
        self, X: np.ndarray, y: np.ndarray, model, cv_folds: int = 10, n_repeats: int = 10, cv_splitter=None
    ) -> Dict[str, float]:
        """
        Analyze model stability across multiple CV runs.

        Args:
            X: Feature matrix
            y: Labels
            model: Model to evaluate
            cv_folds: Number of CV folds (ignored if cv_splitter provided)
            n_repeats: Number of times to repeat CV
            cv_splitter: Optional custom CV splitter

        Returns:
            Dictionary with stability metrics
        """
        accuracies = []
        per_class_accuracies = []

        # Check if using LeaveOneOut
        from sklearn.model_selection import LeaveOneOut
        
        if cv_splitter is not None and isinstance(cv_splitter, LeaveOneOut):
            # LeaveOneOut doesn't support multiple repeats
            n_repeats = 1
            
        for repeat in range(n_repeats):
            if cv_splitter is not None:
                cv = cv_splitter
            else:
                cv = StratifiedKFold(
                    n_splits=cv_folds, shuffle=True, random_state=self.random_state + repeat
                )
            y_pred = cross_val_predict(model, X, y, cv=cv)

            # Overall accuracy
            accuracies.append(accuracy_score(y, y_pred))

            # Per-class accuracy
            cm = confusion_matrix(y, y_pred)
            per_class_acc = np.diag(cm) / cm.sum(axis=1)
            per_class_accuracies.append(per_class_acc)

        # Calculate statistics
        accuracies = np.array(accuracies)
        per_class_accuracies = np.array(per_class_accuracies)

        results = {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "min_accuracy": np.min(accuracies),
            "max_accuracy": np.max(accuracies),
            "95_ci_lower": np.percentile(accuracies, 2.5),
            "95_ci_upper": np.percentile(accuracies, 97.5),
            "per_class_mean": np.mean(per_class_accuracies, axis=0),
            "per_class_std": np.std(per_class_accuracies, axis=0),
        }

        return results

    def paired_t_test(self, scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """
        Perform paired t-test for comparing two methods.

        Args:
            scores1: Scores from first method (e.g., across CV folds)
            scores2: Scores from second method

        Returns:
            Dictionary with test results
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        # Effect size (Cohen's d for paired samples)
        diff = scores1 - scores2
        d = np.mean(diff) / np.std(diff, ddof=1)

        return {
            "mean_score1": np.mean(scores1),
            "mean_score2": np.mean(scores2),
            "mean_difference": np.mean(diff),
            "std_difference": np.std(diff, ddof=1),
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": d,
            "significant": p_value < 0.05,
        }