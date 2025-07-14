"""
Baseline comparison methods for phoneme classification.

Provides traditional baselines to compare against the contrastive learning approach:
- Raw MFCC features + SVM
- Raw MFCC features + Random Forest
- Mel-spectrogram + CNN (simple)
- Hand-crafted acoustic features
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BaselineComparison:
    """Compare contrastive learning against traditional baselines."""

    def __init__(self, random_state: int = 42):
        """
        Initialize baseline comparisons.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def extract_mfcc_features(
        self,
        waveforms: List[torch.Tensor],
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        include_delta: bool = True,
        include_delta_delta: bool = True,
    ) -> np.ndarray:
        """
        Extract MFCC features with optional deltas.

        Args:
            waveforms: List of audio waveforms
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            include_delta: Include first derivatives
            include_delta_delta: Include second derivatives

        Returns:
            Feature matrix [n_samples, n_features]
        """
        all_features = []

        # MFCC transform
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23},
        )

        for waveform in waveforms:
            # Extract MFCCs
            mfcc = mfcc_transform(waveform)

            # Calculate statistics over time
            features = []

            # Mean and std of each coefficient
            features.extend(mfcc.mean(dim=-1).squeeze().numpy())
            features.extend(mfcc.std(dim=-1).squeeze().numpy())

            # Add deltas if requested
            if include_delta:
                delta = torchaudio.functional.compute_deltas(mfcc)
                features.extend(delta.mean(dim=-1).squeeze().numpy())
                features.extend(delta.std(dim=-1).squeeze().numpy())

            if include_delta_delta:
                delta = torchaudio.functional.compute_deltas(mfcc)
                delta_delta = torchaudio.functional.compute_deltas(delta)
                features.extend(delta_delta.mean(dim=-1).squeeze().numpy())
                features.extend(delta_delta.std(dim=-1).squeeze().numpy())

            all_features.append(features)

        return np.array(all_features)

    def extract_spectral_features(
        self, waveforms: List[torch.Tensor], sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract hand-crafted spectral features.

        Features include:
        - Spectral centroid
        - Spectral rolloff
        - Spectral bandwidth
        - Zero crossing rate
        - RMS energy

        Args:
            waveforms: List of audio waveforms
            sample_rate: Audio sample rate

        Returns:
            Feature matrix [n_samples, n_features]
        """
        all_features = []

        for waveform in waveforms:
            waveform = waveform.squeeze()
            features = []

            # Spectral centroid
            spec = torch.stft(
                waveform,
                n_fft=400,
                hop_length=160,
                return_complex=True,
            )
            magnitudes = torch.abs(spec)
            freqs = torch.linspace(0, sample_rate / 2, magnitudes.size(0))

            # Spectral centroid
            centroid = (freqs.unsqueeze(1) * magnitudes).sum(0) / (magnitudes.sum(0) + 1e-10)
            features.extend([centroid.mean().item(), centroid.std().item()])

            # Spectral rolloff (85th percentile)
            cumsum = torch.cumsum(magnitudes, dim=0)
            rolloff_threshold = 0.85 * magnitudes.sum(0)
            rolloff_bins = torch.argmax((cumsum >= rolloff_threshold).float(), dim=0)
            rolloff_freq = freqs[rolloff_bins]
            features.extend([rolloff_freq.mean().item(), rolloff_freq.std().item()])

            # Spectral bandwidth
            normalized_freqs = freqs / (sample_rate / 2)
            bandwidth = torch.sqrt(
                ((normalized_freqs.unsqueeze(1) - centroid) ** 2 * magnitudes).sum(0)
                / (magnitudes.sum(0) + 1e-10)
            )
            features.extend([bandwidth.mean().item(), bandwidth.std().item()])

            # Zero crossing rate
            zero_crossings = torch.sum(torch.diff(torch.sign(waveform)) != 0).float()
            zcr = zero_crossings / len(waveform)
            features.append(zcr.item())

            # RMS energy
            rms = torch.sqrt(torch.mean(waveform**2))
            features.append(rms.item())

            all_features.append(features)

        return np.array(all_features)

    def create_baseline_pipelines(self) -> Dict[str, Pipeline]:
        """
        Create scikit-learn pipelines for baseline models.

        Returns:
            Dictionary of baseline pipelines
        """
        pipelines = {}

        # 1. SVM with RBF kernel
        pipelines["svm_rbf"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", random_state=self.random_state)),
            ]
        )

        # 2. Linear SVM
        pipelines["svm_linear"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="linear", random_state=self.random_state)),
            ]
        )

        # 3. Random Forest
        pipelines["random_forest"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ]
        )

        # 4. SVM with PCA
        pipelines["pca_svm"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95)),  # Keep 95% variance
                ("svm", SVC(kernel="rbf", random_state=self.random_state)),
            ]
        )

        return pipelines

    def tune_baseline_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = "svm_rbf",
        cv_folds: int = 5,
    ) -> Tuple[Pipeline, Dict]:
        """
        Tune hyperparameters for a baseline model.

        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the baseline model
            cv_folds: Number of cross-validation folds

        Returns:
            (best_model, best_params)
        """
        pipelines = self.create_baseline_pipelines()
        pipeline = pipelines[model_name]

        # Define parameter grids
        param_grids = {
            "svm_rbf": {
                "svm__C": [0.1, 1, 10, 100],
                "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            },
            "svm_linear": {
                "svm__C": [0.01, 0.1, 1, 10, 100],
            },
            "random_forest": {
                "rf__n_estimators": [50, 100, 200],
                "rf__max_depth": [None, 10, 20, 30],
                "rf__min_samples_split": [2, 5, 10],
            },
            "pca_svm": {
                "pca__n_components": [20, 50, 100, 0.95, 0.99],
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale", 0.01, 0.1],
            },
        }

        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate_all_baselines(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        tune_hyperparameters: bool = True,
    ) -> Dict[str, Dict]:
        """
        Evaluate all baseline models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary with results for each baseline
        """
        results = {}
        pipelines = self.create_baseline_pipelines()

        for model_name, pipeline in pipelines.items():
            self.logger.info(f"Evaluating baseline: {model_name}")

            if tune_hyperparameters and model_name in ["svm_rbf", "svm_linear"]:
                # Tune only SVM models (faster)
                model, best_params = self.tune_baseline_model(X_train, y_train, model_name)
                self.logger.info(f"Best params for {model_name}: {best_params}")
            else:
                model = pipeline
                model.fit(X_train, y_train)

            # Evaluate
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

            results[model_name] = {
                "train_accuracy": accuracy_score(y_train, train_pred),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "cv_accuracy_mean": np.mean(cv_scores),
                "cv_accuracy_std": np.std(cv_scores),
                "model": model,
            }

        return results

    def compare_feature_sets(
        self,
        waveforms_train: List[torch.Tensor],
        y_train: np.ndarray,
        waveforms_test: List[torch.Tensor],
        y_test: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict[str, Dict]:
        """
        Compare different feature extraction methods.

        Args:
            waveforms_train: Training waveforms
            y_train: Training labels
            waveforms_test: Test waveforms
            y_test: Test labels
            sample_rate: Audio sample rate

        Returns:
            Dictionary with results for each feature set
        """
        results = {}

        # 1. Basic MFCC (13 coefficients)
        self.logger.info("Extracting basic MFCC features...")
        X_train_mfcc = self.extract_mfcc_features(
            waveforms_train, sample_rate, n_mfcc=13, include_delta=False, include_delta_delta=False
        )
        X_test_mfcc = self.extract_mfcc_features(
            waveforms_test, sample_rate, n_mfcc=13, include_delta=False, include_delta_delta=False
        )

        # 2. MFCC with deltas
        self.logger.info("Extracting MFCC + delta features...")
        X_train_mfcc_delta = self.extract_mfcc_features(
            waveforms_train, sample_rate, n_mfcc=13, include_delta=True, include_delta_delta=True
        )
        X_test_mfcc_delta = self.extract_mfcc_features(
            waveforms_test, sample_rate, n_mfcc=13, include_delta=True, include_delta_delta=True
        )

        # 3. Spectral features
        self.logger.info("Extracting spectral features...")
        X_train_spectral = self.extract_spectral_features(waveforms_train, sample_rate)
        X_test_spectral = self.extract_spectral_features(waveforms_test, sample_rate)

        # 4. Combined features
        self.logger.info("Combining features...")
        X_train_combined = np.hstack([X_train_mfcc_delta, X_train_spectral])
        X_test_combined = np.hstack([X_test_mfcc_delta, X_test_spectral])

        # Evaluate each feature set
        feature_sets = {
            "mfcc_basic": (X_train_mfcc, X_test_mfcc),
            "mfcc_delta": (X_train_mfcc_delta, X_test_mfcc_delta),
            "spectral": (X_train_spectral, X_test_spectral),
            "combined": (X_train_combined, X_test_combined),
        }

        # Use linear SVM for all comparisons
        for feature_name, (X_train, X_test) in feature_sets.items():
            self.logger.info(f"Evaluating {feature_name} features...")

            # Train model
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="linear", random_state=self.random_state)),
                ]
            )
            model.fit(X_train, y_train)

            # Evaluate
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

            results[feature_name] = {
                "n_features": X_train.shape[1],
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "cv_accuracy_mean": np.mean(cv_scores),
                "cv_accuracy_std": np.std(cv_scores),
            }

        return results

    def generate_baseline_report(
        self,
        baseline_results: Dict[str, Dict],
        contrastive_results: Dict[str, float],
    ) -> str:
        """
        Generate a comparison report between baselines and contrastive learning.

        Args:
            baseline_results: Results from baseline models
            contrastive_results: Results from contrastive learning model

        Returns:
            Formatted report string
        """
        report = "## Baseline Comparison Report\n\n"

        # Add contrastive results to comparison
        all_results = {**baseline_results}
        all_results["contrastive_learning"] = contrastive_results

        # Sort by test accuracy
        sorted_models = sorted(
            all_results.items(),
            key=lambda x: x[1].get("test_accuracy", 0),
            reverse=True,
        )

        report += "### Model Performance Ranking\n\n"
        report += "| Model | Test Acc | CV Acc (mean±std) | Train Acc |\n"
        report += "|-------|----------|-------------------|------------|\n"

        for model_name, results in sorted_models:
            test_acc = results.get("test_accuracy", 0)
            train_acc = results.get("train_accuracy", 0)
            cv_mean = results.get("cv_accuracy_mean", 0)
            cv_std = results.get("cv_accuracy_std", 0)

            report += f"| {model_name} | {test_acc:.3f} | {cv_mean:.3f}±{cv_std:.3f} | {train_acc:.3f} |\n"

        # Best baseline vs contrastive
        best_baseline = (
            sorted_models[0] if sorted_models[0][0] != "contrastive_learning" else sorted_models[1]
        )

        report += "\n### Key Findings\n\n"
        report += f"- Best baseline: {best_baseline[0]} ({best_baseline[1]['test_accuracy']:.3f})\n"
        report += f"- Contrastive learning: {contrastive_results.get('test_accuracy', 0):.3f}\n"

        improvement = (
            (contrastive_results.get("test_accuracy", 0) - best_baseline[1]["test_accuracy"])
            / best_baseline[1]["test_accuracy"]
            * 100
        )

        report += f"- Relative improvement: {improvement:+.1f}%\n"

        return report
