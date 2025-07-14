#!/usr/bin/env python3
"""
Quick ablation study to systematically test feature components.
Runs shorter experiments to get key insights efficiently.
"""

import json
import subprocess
import time
from pathlib import Path

import pandas as pd


def run_experiment(config_overrides: dict, name: str, epochs: int = 50):
    """Run a single experiment with given configuration."""

    print(f"\n🚀 Running {name}")
    print(f"Config: {config_overrides}")

    # Build command
    cmd = ["uv", "run", "scripts/train.py"]

    # Add config overrides
    for key, value in config_overrides.items():
        cmd.extend([f"{key}={value}"])

    # Add epochs
    cmd.extend([f"training.epochs={epochs}"])

    # Run experiment
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    if result.returncode != 0:
        print(f"❌ {name} failed: {result.stderr}")
        return None

    # Find output directory
    output_dirs = list(Path("outputs").glob("2025-*/*"))
    if not output_dirs:
        print(f"❌ No output directory found for {name}")
        return None

    # Get latest directory
    latest_dir = max(output_dirs, key=lambda p: p.stat().st_mtime)

    # Load metrics
    metrics_file = latest_dir / "metrics.json"
    if not metrics_file.exists():
        print(f"❌ No metrics file found for {name}")
        return None

    with open(metrics_file) as f:
        metrics = json.load(f)

    # Extract final results
    final_results = {
        "name": name,
        "config": config_overrides,
        "output_dir": str(latest_dir),
        "duration_minutes": (end_time - start_time) / 60,
        "final_train_loss": metrics["train_loss"][-1],
        "final_val_loss": metrics["val_loss"][-1],
        "final_linear_acc": metrics["val_linear_accuracy"][-1]
        if "val_linear_accuracy" in metrics
        else None,
        "final_rf_acc": metrics["val_rf_accuracy"][-1] if "val_rf_accuracy" in metrics else None,
        "peak_linear_acc": max(metrics["val_linear_accuracy"])
        if "val_linear_accuracy" in metrics
        else None,
        "peak_rf_acc": max(metrics["val_rf_accuracy"]) if "val_rf_accuracy" in metrics else None,
    }

    print(f"✅ {name} completed in {final_results['duration_minutes']:.1f}m")
    if final_results["final_rf_acc"]:
        print(f"   RF Accuracy: {final_results['final_rf_acc']:.1%}")
    if final_results["final_linear_acc"]:
        print(f"   Linear Accuracy: {final_results['final_linear_acc']:.1%}")

    return final_results


def main():
    """Run systematic ablation study."""

    print("🔬 SYSTEMATIC ABLATION STUDY")
    print("=" * 50)

    # Define experiments
    experiments = [
        {
            "name": "Baseline (40 MFCC)",
            "config": {
                "model": "tcn",
                "data.feature_extractor.mfcc_params.n_mfcc": 40,
                "data.feature_extractor.mfcc_params.add_delta": "false",
                "data.feature_extractor.mfcc_params.add_delta_delta": "false",
                "model.in_channels": 40,
            },
        },
        {
            "name": "More MFCCs (60)",
            "config": {
                "model": "tcn",
                "data.feature_extractor.mfcc_params.n_mfcc": 60,
                "data.feature_extractor.mfcc_params.add_delta": "false",
                "data.feature_extractor.mfcc_params.add_delta_delta": "false",
                "model.in_channels": 60,
            },
        },
        {
            "name": "Delta Only (40+40)",
            "config": {
                "model": "tcn",
                "data.feature_extractor.mfcc_params.n_mfcc": 40,
                "data.feature_extractor.mfcc_params.add_delta": "true",
                "data.feature_extractor.mfcc_params.add_delta_delta": "false",
                "model.in_channels": 80,
            },
        },
        {
            "name": "Delta-Delta Only (40+40)",
            "config": {
                "model": "tcn",
                "data.feature_extractor.mfcc_params.n_mfcc": 40,
                "data.feature_extractor.mfcc_params.add_delta": "false",
                "data.feature_extractor.mfcc_params.add_delta_delta": "true",
                "model.in_channels": 80,
            },
        },
        {
            "name": "Full Enhanced (60+60+60)",
            "config": {
                "model": "tcn",
                "data.feature_extractor.mfcc_params.n_mfcc": 60,
                "data.feature_extractor.mfcc_params.add_delta": "true",
                "data.feature_extractor.mfcc_params.add_delta_delta": "true",
                "model.in_channels": 180,
            },
        },
    ]

    # Run experiments
    results = []
    for exp in experiments:
        result = run_experiment(exp["config"], exp["name"], epochs=50)
        if result:
            results.append(result)

        # Brief pause between experiments
        time.sleep(5)

    # Create results summary
    if results:
        print("\n📊 ABLATION STUDY RESULTS")
        print("=" * 80)

        # Create DataFrame for easy analysis
        df = pd.DataFrame(results)

        # Sort by RF accuracy
        df = df.sort_values("final_rf_acc", ascending=False)

        print(f"{'Experiment':<25} {'RF Acc':<8} {'Linear':<8} {'Duration':<8}")
        print("-" * 60)

        for _, row in df.iterrows():
            rf_acc = f"{row['final_rf_acc']:.1%}" if row["final_rf_acc"] else "N/A"
            linear_acc = f"{row['final_linear_acc']:.1%}" if row["final_linear_acc"] else "N/A"
            duration = f"{row['duration_minutes']:.1f}m"

            print(f"{row['name']:<25} {rf_acc:<8} {linear_acc:<8} {duration:<8}")

        # Calculate improvements
        baseline_rf = df[df["name"] == "Baseline (40 MFCC)"]["final_rf_acc"].iloc[0]

        print("\n🚀 IMPROVEMENTS OVER BASELINE")
        print("-" * 50)

        for _, row in df.iterrows():
            if row["name"] == "Baseline (40 MFCC)":
                continue

            if row["final_rf_acc"] and baseline_rf:
                improvement = row["final_rf_acc"] - baseline_rf
                print(f"{row['name']:<25} {improvement:+.1%} ({improvement * 100:+.1f} points)")

        # Save detailed results
        results_file = Path("ablation_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Detailed results saved to: {results_file}")

        # Key findings
        best_exp = df.iloc[0]
        print(f"\n🏆 BEST PERFORMING: {best_exp['name']}")
        print(f"   RF Accuracy: {best_exp['final_rf_acc']:.1%}")
        print(f"   Linear Accuracy: {best_exp['final_linear_acc']:.1%}")

        # Feature importance ranking
        feature_ranking = []
        for _, row in df.iterrows():
            if row["final_rf_acc"] and baseline_rf:
                improvement = row["final_rf_acc"] - baseline_rf
                feature_ranking.append((row["name"], improvement))

        feature_ranking.sort(key=lambda x: x[1], reverse=True)

        print("\n🔍 FEATURE IMPORTANCE RANKING:")
        for i, (name, improvement) in enumerate(feature_ranking, 1):
            print(f"{i}. {name}: {improvement:+.1%}")


if __name__ == "__main__":
    main()
