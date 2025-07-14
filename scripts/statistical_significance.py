#!/usr/bin/env python3
"""
Statistical significance testing for phoneme classification improvements.
Implements McNemar's test, bootstrap confidence intervals, and effect size analysis.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def mcnemar_test_manual(table):
    """Manual implementation of McNemar's test."""
    # table is 2x2: [[a, b], [c, d]]
    # We test b vs c (disagreements)
    b = table[0][1]
    c = table[1][0]

    if b + c == 0:
        return 0.0, 1.0  # No disagreements

    # Chi-square test with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value


class StatisticalTester:
    """Handles statistical significance testing for model comparisons."""

    def __init__(self):
        self.results = {}

    def run_multiple_seeds(self, config: Dict, n_seeds: int = 5, epochs: int = 80) -> List[Dict]:
        """Run experiments with multiple random seeds."""

        print(f"🎲 Running {n_seeds} experiments with different seeds...")

        results = []
        seeds = [42, 123, 456, 789, 999][:n_seeds]

        for i, seed in enumerate(seeds, 1):
            print(f"\n📊 Experiment {i}/{n_seeds} (seed={seed})")

            # Build command
            cmd = ["uv", "run", "scripts/train.py"]

            # Add config
            for key, value in config.items():
                cmd.append(f"{key}={value}")

            # Add seed and epochs
            cmd.extend([f"experiment.seed={seed}", f"training.epochs={epochs}"])

            # Run experiment
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ Seed {seed} failed: {result.stderr}")
                continue

            # Get latest output directory
            output_dirs = list(Path("outputs").glob("2025-*/*"))
            if not output_dirs:
                continue

            latest_dir = max(output_dirs, key=lambda p: p.stat().st_mtime)

            # Load metrics
            metrics_file = latest_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

                result_data = {
                    "seed": seed,
                    "output_dir": str(latest_dir),
                    "final_rf_acc": metrics["val_rf_accuracy"][-1],
                    "final_linear_acc": metrics["val_linear_accuracy"][-1],
                    "peak_rf_acc": max(metrics["val_rf_accuracy"]),
                    "peak_linear_acc": max(metrics["val_linear_accuracy"]),
                    "final_loss": metrics["val_loss"][-1],
                }

                results.append(result_data)
                print(f"✅ Seed {seed}: RF={result_data['final_rf_acc']:.1%}")

        return results

    def mcnemar_test(
        self, predictions_a: np.ndarray, predictions_b: np.ndarray, true_labels: np.ndarray
    ) -> Dict:
        """Perform McNemar's test for comparing two classifiers."""

        # Create contingency table
        correct_a = predictions_a == true_labels
        correct_b = predictions_b == true_labels

        # McNemar table: [both_wrong, a_right_b_wrong, a_wrong_b_right, both_right]
        both_wrong = np.sum(~correct_a & ~correct_b)
        a_right_b_wrong = np.sum(correct_a & ~correct_b)
        a_wrong_b_right = np.sum(~correct_a & correct_b)
        both_right = np.sum(correct_a & correct_b)

        # McNemar test on disagreements
        if a_right_b_wrong + a_wrong_b_right == 0:
            statistic, p_value = 0.0, 1.0  # No disagreements
        else:
            statistic, p_value = mcnemar_test_manual(
                [[both_wrong, a_wrong_b_right], [a_right_b_wrong, both_right]]
            )

        return {
            "statistic": float(statistic) if a_right_b_wrong + a_wrong_b_right > 0 else 0.0,
            "p_value": float(p_value),
            "contingency_table": {
                "both_wrong": int(both_wrong),
                "a_right_b_wrong": int(a_right_b_wrong),
                "a_wrong_b_right": int(a_wrong_b_right),
                "both_right": int(both_right),
            },
            "significant": p_value < 0.05,
        }

    def bootstrap_confidence_interval(
        self, scores: List[float], n_bootstrap: int = 1000, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for accuracy scores."""

        scores = np.array(scores)
        n_samples = len(scores)

        bootstrap_scores = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
            bootstrap_scores.append(np.mean(bootstrap_sample))

        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)

        return ci_lower, ci_upper

    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""

        group1, group2 = np.array(group1), np.array(group2)

        # Calculate means
        mean1, mean2 = np.mean(group1), np.mean(group2)

        # Calculate pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2)
        )

        # Cohen's d
        if pooled_std == 0:
            return 0.0

        d = (mean1 - mean2) / pooled_std
        return d

    def analyze_significance(
        self, baseline_results: List[Dict], enhanced_results: List[Dict]
    ) -> Dict:
        """Comprehensive statistical analysis comparing two approaches."""

        print("\n📈 STATISTICAL SIGNIFICANCE ANALYSIS")
        print("=" * 60)

        # Extract RF accuracies
        baseline_rf = [r["final_rf_acc"] for r in baseline_results]
        enhanced_rf = [r["final_rf_acc"] for r in enhanced_results]

        # Extract linear accuracies
        baseline_linear = [r["final_linear_acc"] for r in baseline_results]
        enhanced_linear = [r["final_linear_acc"] for r in enhanced_results]

        # Basic statistics
        baseline_rf_mean = np.mean(baseline_rf)
        enhanced_rf_mean = np.mean(enhanced_rf)
        baseline_linear_mean = np.mean(baseline_linear)
        enhanced_linear_mean = np.mean(enhanced_linear)

        print(f"Baseline RF Accuracy: {baseline_rf_mean:.1%} ± {np.std(baseline_rf):.1%}")
        print(f"Enhanced RF Accuracy: {enhanced_rf_mean:.1%} ± {np.std(enhanced_rf):.1%}")
        print(f"Improvement: {enhanced_rf_mean - baseline_rf_mean:+.1%}")

        # Statistical tests
        analysis = {}

        # 1. T-test for RF accuracy
        rf_tstat, rf_pvalue = stats.ttest_ind(enhanced_rf, baseline_rf)
        analysis["rf_ttest"] = {
            "statistic": float(rf_tstat),
            "p_value": float(rf_pvalue),
            "significant": rf_pvalue < 0.05,
        }

        # 2. T-test for linear accuracy
        linear_tstat, linear_pvalue = stats.ttest_ind(enhanced_linear, baseline_linear)
        analysis["linear_ttest"] = {
            "statistic": float(linear_tstat),
            "p_value": float(linear_pvalue),
            "significant": linear_pvalue < 0.05,
        }

        # 3. Effect sizes
        rf_effect_size = self.cohens_d(enhanced_rf, baseline_rf)
        linear_effect_size = self.cohens_d(enhanced_linear, baseline_linear)

        analysis["effect_sizes"] = {
            "rf_cohens_d": rf_effect_size,
            "linear_cohens_d": linear_effect_size,
        }

        # 4. Bootstrap confidence intervals
        rf_ci_baseline = self.bootstrap_confidence_interval(baseline_rf)
        rf_ci_enhanced = self.bootstrap_confidence_interval(enhanced_rf)
        linear_ci_baseline = self.bootstrap_confidence_interval(baseline_linear)
        linear_ci_enhanced = self.bootstrap_confidence_interval(enhanced_linear)

        analysis["confidence_intervals"] = {
            "baseline_rf_95ci": rf_ci_baseline,
            "enhanced_rf_95ci": rf_ci_enhanced,
            "baseline_linear_95ci": linear_ci_baseline,
            "enhanced_linear_95ci": linear_ci_enhanced,
        }

        # Print results
        print("\n🧪 STATISTICAL TEST RESULTS:")
        print(
            f"RF Accuracy t-test: t={rf_tstat:.3f}, p={rf_pvalue:.4f} {'✅ SIGNIFICANT' if rf_pvalue < 0.05 else '❌ NOT SIGNIFICANT'}"
        )
        print(
            f"Linear Accuracy t-test: t={linear_tstat:.3f}, p={linear_pvalue:.4f} {'✅ SIGNIFICANT' if linear_pvalue < 0.05 else '❌ NOT SIGNIFICANT'}"
        )

        print("\n📏 EFFECT SIZES (Cohen's d):")
        print(
            f"RF Accuracy: d={rf_effect_size:.3f} ({self._interpret_effect_size(rf_effect_size)})"
        )
        print(
            f"Linear Accuracy: d={linear_effect_size:.3f} ({self._interpret_effect_size(linear_effect_size)})"
        )

        print("\n🎯 CONFIDENCE INTERVALS (95%):")
        print(f"Baseline RF: [{rf_ci_baseline[0]:.1%}, {rf_ci_baseline[1]:.1%}]")
        print(f"Enhanced RF: [{rf_ci_enhanced[0]:.1%}, {rf_ci_enhanced[1]:.1%}]")
        print(f"Overlap: {'❌ NO' if rf_ci_enhanced[0] > rf_ci_baseline[1] else '⚠️  YES'}")

        # Summary assessment
        analysis["summary"] = {
            "rf_improvement_significant": rf_pvalue < 0.05,
            "linear_improvement_significant": linear_pvalue < 0.05,
            "rf_large_effect": abs(rf_effect_size) > 0.8,
            "linear_large_effect": abs(linear_effect_size) > 0.8,
            "confidence_intervals_separate": rf_ci_enhanced[0] > rf_ci_baseline[1],
        }

        return analysis

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def create_significance_plot(
        self,
        baseline_results: List[Dict],
        enhanced_results: List[Dict],
        save_path: str = "statistical_significance.png",
    ):
        """Create visualization of statistical significance results."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data
        baseline_rf = [r["final_rf_acc"] for r in baseline_results]
        enhanced_rf = [r["final_rf_acc"] for r in enhanced_results]
        baseline_linear = [r["final_linear_acc"] for r in baseline_results]
        enhanced_linear = [r["final_linear_acc"] for r in enhanced_results]

        # 1. RF Accuracy comparison
        axes[0, 0].boxplot([baseline_rf, enhanced_rf], labels=["Baseline", "Enhanced"])
        axes[0, 0].set_title("RF Accuracy Distribution")
        axes[0, 0].set_ylabel("Accuracy")

        # 2. Linear Accuracy comparison
        axes[0, 1].boxplot([baseline_linear, enhanced_linear], labels=["Baseline", "Enhanced"])
        axes[0, 1].set_title("Linear Accuracy Distribution")
        axes[0, 1].set_ylabel("Accuracy")

        # 3. Confidence intervals
        means_rf = [np.mean(baseline_rf), np.mean(enhanced_rf)]
        ci_rf_baseline = self.bootstrap_confidence_interval(baseline_rf)
        ci_rf_enhanced = self.bootstrap_confidence_interval(enhanced_rf)

        axes[1, 0].errorbar(
            [0, 1],
            means_rf,
            yerr=[
                [means_rf[0] - ci_rf_baseline[0], means_rf[1] - ci_rf_enhanced[0]],
                [ci_rf_baseline[1] - means_rf[0], ci_rf_enhanced[1] - means_rf[1]],
            ],
            fmt="o",
            capsize=5,
            capthick=2,
        )
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_xticklabels(["Baseline", "Enhanced"])
        axes[1, 0].set_title("RF Accuracy 95% Confidence Intervals")
        axes[1, 0].set_ylabel("Accuracy")

        # 4. Effect size visualization
        rf_effect = self.cohens_d(enhanced_rf, baseline_rf)
        linear_effect = self.cohens_d(enhanced_linear, baseline_linear)

        effects = [rf_effect, linear_effect]
        labels = ["RF Accuracy", "Linear Accuracy"]
        colors = ["green" if e > 0.8 else "orange" if e > 0.5 else "red" for e in effects]

        bars = axes[1, 1].bar(labels, effects, color=colors, alpha=0.7)
        axes[1, 1].set_title("Effect Sizes (Cohen's d)")
        axes[1, 1].set_ylabel("Effect Size")
        axes[1, 1].axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Large effect")
        axes[1, 1].axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Medium effect")
        axes[1, 1].legend()

        # Add value labels
        for bar, effect in zip(bars, effects):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                effect + 0.05,
                f"d={effect:.2f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n📊 Statistical significance plot saved: {save_path}")

        return fig


def main():
    """Run comprehensive statistical significance analysis."""

    tester = StatisticalTester()

    print("🔬 STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 70)

    # Configuration for baseline vs enhanced
    baseline_config = {
        "model": "tcn",
        "data.feature_extractor.mfcc_params.n_mfcc": 40,
        "data.feature_extractor.mfcc_params.add_delta": "false",
        "data.feature_extractor.mfcc_params.add_delta_delta": "false",
        "model.in_channels": 40,
    }

    enhanced_config = {
        "model": "tcn",
        "data.feature_extractor.mfcc_params.n_mfcc": 60,
        "data.feature_extractor.mfcc_params.add_delta": "true",
        "data.feature_extractor.mfcc_params.add_delta_delta": "true",
        "model.in_channels": 180,
    }

    # Run multiple seeds for both configurations
    print("\n1️⃣ Running baseline experiments...")
    baseline_results = tester.run_multiple_seeds(baseline_config, n_seeds=5, epochs=60)

    print("\n2️⃣ Running enhanced experiments...")
    enhanced_results = tester.run_multiple_seeds(enhanced_config, n_seeds=5, epochs=60)

    if len(baseline_results) < 3 or len(enhanced_results) < 3:
        print("❌ Insufficient successful runs for statistical testing")
        return

    # Perform statistical analysis
    analysis = tester.analyze_significance(baseline_results, enhanced_results)

    # Create visualization
    tester.create_significance_plot(baseline_results, enhanced_results)

    # Save detailed results
    results = {
        "baseline_results": baseline_results,
        "enhanced_results": enhanced_results,
        "statistical_analysis": analysis,
        "timestamp": str(Path().cwd()),
    }

    with open("statistical_significance_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n💾 Complete results saved: statistical_significance_results.json")

    # Final summary
    summary = analysis["summary"]
    print("\n🎯 FINAL STATISTICAL ASSESSMENT:")
    print(
        f"RF Improvement Significant: {'✅ YES' if summary['rf_improvement_significant'] else '❌ NO'}"
    )
    print(f"Large Effect Size (RF): {'✅ YES' if summary['rf_large_effect'] else '❌ NO'}")
    print(
        f"Confidence Intervals Separate: {'✅ YES' if summary['confidence_intervals_separate'] else '❌ NO'}"
    )

    if all(
        [
            summary["rf_improvement_significant"],
            summary["rf_large_effect"],
            summary["confidence_intervals_separate"],
        ]
    ):
        print(
            "\n🏆 CONCLUSION: Enhanced features show STATISTICALLY SIGNIFICANT improvement with LARGE effect size!"
        )
    else:
        print("\n⚠️  CONCLUSION: Statistical evidence is mixed or insufficient.")


if __name__ == "__main__":
    main()
