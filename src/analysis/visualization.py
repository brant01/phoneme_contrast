"""
Publication-ready visualization module.

Creates high-quality figures suitable for academic papers.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage


class PublicationVisualizer:
    """Create publication-quality visualizations."""

    # Publication-ready settings
    STYLE_CONFIG = {
        "figure.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    }

    def __init__(self, style: str = "paper"):
        """
        Initialize visualizer with publication settings.

        Args:
            style: Visual style ("paper", "presentation", "poster")
        """
        self.style = style
        self.logger = logging.getLogger(__name__)

        # Apply style settings
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(self.STYLE_CONFIG)

        # Adjust for different contexts
        if style == "presentation":
            plt.rcParams.update({"font.size": 14, "axes.labelsize": 14})
        elif style == "poster":
            plt.rcParams.update({"font.size": 18, "axes.labelsize": 18})

    def plot_embedding_space(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        phoneme_names: List[str],
        feature_groups: Optional[Dict[str, List[str]]] = None,
        title: str = "Phoneme Embedding Space",
        figsize: Tuple[float, float] = (8, 6),
    ) -> plt.Figure:
        """
        Create publication-quality t-SNE visualization.

        Args:
            embeddings: 2D embeddings (already reduced)
            labels: Numeric labels
            phoneme_names: List of phoneme strings
            feature_groups: Optional grouping by features
            title: Figure title
            figsize: Figure size in inches

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # If feature groups provided, use them for coloring
        if feature_groups:
            # Create color map for features
            feature_colors = {}
            color_palette = sns.color_palette("husl", len(feature_groups))

            for idx, (feature_name, phonemes) in enumerate(feature_groups.items()):
                for phoneme in phonemes:
                    feature_colors[phoneme] = color_palette[idx]

            # Plot by feature groups
            for feature_name, phonemes in feature_groups.items():
                mask = np.array([p in phonemes for p in phoneme_names])
                if np.any(mask):
                    ax.scatter(
                        embeddings[mask, 0],
                        embeddings[mask, 1],
                        c=[feature_colors[p] for p in np.array(phoneme_names)[mask]],
                        label=feature_name,
                        alpha=0.7,
                        s=50,
                        edgecolors="black",
                        linewidth=0.5,
                    )
        else:
            # Use default coloring by label
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels,
                cmap="tab20",
                alpha=0.7,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

        # Add select phoneme labels for clarity
        # Only label a subset to avoid overcrowding
        unique_labels = np.unique(labels)
        for label in unique_labels[::3]:  # Every 3rd label
            mask = labels == label
            if np.any(mask):
                center = embeddings[mask].mean(axis=0)
                phoneme = phoneme_names[np.where(labels == label)[0][0]]
                ax.annotate(
                    phoneme,
                    center,
                    fontsize=8,
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title(title)

        if feature_groups:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        return fig

    def plot_performance_comparison(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metric: str = "test_accuracy",
        title: str = "Model Performance Comparison",
        figsize: Tuple[float, float] = (8, 5),
    ) -> plt.Figure:
        """
        Create bar plot comparing model performances with confidence intervals.

        Args:
            results_dict: Dictionary mapping model names to results
            metric: Metric to plot
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Extract data
        models = []
        values = []
        errors = []

        for model_name, results in results_dict.items():
            if metric in results:
                models.append(model_name.replace("_", " ").title())
                values.append(results[metric])

                # Try to get confidence interval or std
                if f"{metric}_ci_lower" in results and f"{metric}_ci_upper" in results:
                    lower = results[f"{metric}_ci_lower"]
                    upper = results[f"{metric}_ci_upper"]
                    errors.append([results[metric] - lower, upper - results[metric]])
                elif f"{metric}_std" in results:
                    errors.append(results[f"{metric}_std"])
                else:
                    errors.append(0)

        # Create bar plot
        y_pos = np.arange(len(models))
        bars = ax.barh(
            y_pos,
            values,
            xerr=errors if any(errors) else None,
            capsize=5,
            color=sns.color_palette("husl", len(models)),
        )

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(title)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=8,
            )

        # Set x-axis limits
        ax.set_xlim(0, max(values) * 1.15)

        plt.tight_layout()
        return fig

    def plot_confusion_matrix_publication(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
        title: str = "Confusion Matrix",
        figsize: Tuple[float, float] = (10, 8),
        cmap: str = "Blues",
    ) -> plt.Figure:
        """
        Create publication-quality confusion matrix.

        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize by row
            title: Figure title
            figsize: Figure size
            cmap: Colormap

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize if requested
        if normalize:
            cm = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            cm = confusion_matrix
            fmt = "d"

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            square=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8},
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        # Rotate labels if many classes
        if len(class_names) > 20:
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)

        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        feature_scores: Dict[str, float],
        title: str = "Phonetic Feature Importance",
        figsize: Tuple[float, float] = (8, 6),
    ) -> plt.Figure:
        """
        Plot feature importance or separability scores.

        Args:
            feature_scores: Dictionary mapping features to scores
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort features by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        scores = [f[1] for f in sorted_features]

        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, scores, color=sns.color_palette("viridis", len(features)))

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Separability Score")
        ax.set_title(title)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        return fig

    def plot_speaker_invariance_results(
        self,
        invariance_results: Dict[str, float],
        title: str = "Speaker Invariance Analysis",
        figsize: Tuple[float, float] = (10, 6),
    ) -> plt.Figure:
        """
        Visualize speaker invariance metrics.

        Args:
            invariance_results: Dictionary of invariance metrics
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title)

        # 1. Gender classification accuracy (should be low)
        ax = axes[0, 0]
        gender_acc = invariance_results.get("gender_classification_accuracy", 0.5)
        ax.bar(
            ["Gender\nClassification", "Chance"], [gender_acc, 0.5], color=["#e74c3c", "#95a5a6"]
        )
        ax.set_ylabel("Accuracy")
        ax.set_title("Gender Classification from Embeddings")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.5)

        # 2. Cross-gender transfer
        ax = axes[0, 1]
        if "cross_gender_accuracy_m2f" in invariance_results:
            transfer_data = [
                invariance_results.get("cross_gender_accuracy_m2f", 0),
                invariance_results.get("cross_gender_accuracy_f2m", 0),
            ]
            ax.bar(["Male→Female", "Female→Male"], transfer_data, color="#3498db")
            ax.set_ylabel("Accuracy")
            ax.set_title("Cross-Gender Transfer Learning")
            ax.set_ylim(0, 1)

        # 3. Within-phoneme gender distance
        ax = axes[1, 0]
        within_dist = invariance_results.get("mean_within_phoneme_gender_distance", 0)
        between_dist = invariance_results.get("mean_between_phoneme_distance", 1)  # Placeholder
        ax.bar(
            ["Within-Phoneme\n(Different Gender)", "Between-Phoneme"],
            [within_dist, between_dist],
            color=["#2ecc71", "#e67e22"],
        )
        ax.set_ylabel("Distance")
        ax.set_title("Embedding Distances")

        # 4. Summary metrics
        ax = axes[1, 1]
        ax.axis("off")

        # Create summary text
        summary_text = "Summary Metrics:\n\n"
        summary_text += f"Gender Clustering Score: {invariance_results.get('gender_clustering_silhouette', 0):.3f}\n"
        summary_text += f"Variance Ratio: {invariance_results.get('variance_ratio', 0):.3f}\n"

        if "gender_effect_p_value" in invariance_results:
            p_val = invariance_results["gender_effect_p_value"]
            sig = "Yes" if p_val < 0.05 else "No"
            summary_text += f"Gender Effect Significant: {sig} (p={p_val:.3f})"

        ax.text(
            0.1,
            0.5,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
        )

        plt.tight_layout()
        return fig

    def plot_hierarchical_clustering(
        self,
        embeddings: np.ndarray,
        phoneme_names: List[str],
        title: str = "Phoneme Hierarchical Clustering",
        figsize: Tuple[float, float] = (12, 8),
    ) -> plt.Figure:
        """
        Create dendrogram showing phoneme relationships.

        Args:
            embeddings: Mean embeddings per phoneme
            phoneme_names: Phoneme labels
            title: Figure title
            figsize: Figure size

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Compute linkage
        Z = linkage(embeddings, method="ward")

        # Create dendrogram
        dendrogram(
            Z,
            labels=phoneme_names,
            ax=ax,
            leaf_rotation=90,
            leaf_font_size=10,
        )

        ax.set_title(title)
        ax.set_xlabel("Phoneme")
        ax.set_ylabel("Distance")

        plt.tight_layout()
        return fig

    def create_multi_panel_figure(
        self,
        panels: List[plt.Figure],
        labels: List[str],
        title: str = "Comprehensive Analysis",
        figsize: Tuple[float, float] = (12, 10),
    ) -> plt.Figure:
        """
        Combine multiple figures into a single multi-panel figure.

        Args:
            panels: List of individual figures
            labels: Panel labels (A, B, C, etc.)
            title: Overall figure title
            figsize: Figure size

        Returns:
            Combined figure
        """
        # This is a placeholder - in practice you'd arrange subplots
        # For now, return first panel
        if panels:
            return panels[0]
        return plt.figure(figsize=figsize)

    def save_publication_figure(
        self,
        fig: plt.Figure,
        filename: str,
        dpi: int = 300,
        formats: List[str] = ["pdf", "png"],
    ) -> None:
        """
        Save figure in publication-ready formats.

        Args:
            fig: Figure to save
            filename: Base filename (without extension)
            dpi: Resolution for raster formats
            formats: List of formats to save
        """
        for fmt in formats:
            output_path = f"{filename}.{fmt}"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format=fmt)
            self.logger.info(f"Saved figure: {output_path}")
