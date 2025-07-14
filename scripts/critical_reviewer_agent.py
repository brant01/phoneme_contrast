#!/usr/bin/env python3
"""
Critical Reviewer Agent for manuscript sections.
Provides detailed, constructive criticism to improve publication quality.
"""

import sys
from pathlib import Path
from typing import Dict, List


class CriticalReviewer:
    """Acts as a critical reviewer for manuscript sections."""

    def __init__(self):
        self.review_criteria = {
            "methods": [
                "technical_completeness",
                "reproducibility",
                "statistical_rigor",
                "clarity_and_precision",
                "methodological_soundness",
                "ethical_considerations",
            ],
            "results": [
                "data_presentation",
                "statistical_reporting",
                "figure_quality",
                "completeness",
                "objectivity",
                "significance_interpretation",
            ],
            "discussion": [
                "interpretation_validity",
                "limitation_acknowledgment",
                "broader_implications",
                "future_directions",
                "literature_integration",
                "conclusion_support",
            ],
        }

    def review_methods_section(self, content: str) -> Dict[str, List[str]]:
        """Critically review the Methods section."""

        feedback = {
            "strengths": [],
            "critical_issues": [],
            "suggestions": [],
            "missing_elements": [],
        }

        # Analyze content for common issues
        lines = content.lower()

        # Check for technical completeness
        if "hyperparameter" in lines and "grid search" not in lines:
            feedback["critical_issues"].append(
                "Hyperparameter selection not justified - were these values optimized or arbitrarily chosen?"
            )

        if "cross-validation" in lines:
            feedback["strengths"].append("Appropriate use of cross-validation methodology")

        if "statistical significance" in lines:
            feedback["strengths"].append("Inclusion of statistical significance testing")

        # Check for reproducibility issues
        if "seed" in lines:
            feedback["strengths"].append("Random seed control for reproducibility")
        else:
            feedback["critical_issues"].append(
                "No mention of random seed control - results may not be reproducible"
            )

        # Dataset size concerns
        if "126 samples" in content and "limitation" not in lines:
            feedback["critical_issues"].append(
                "Small dataset size (126 samples) is a significant limitation that should be explicitly acknowledged"
            )

        # Statistical power analysis
        if "power analysis" not in lines and "effect size" in lines:
            feedback["suggestions"].append(
                "Consider adding statistical power analysis to justify sample size adequacy"
            )

        # Implementation details
        if "github" not in lines and "code availability" not in lines:
            feedback["missing_elements"].append(
                "Code availability statement missing - essential for reproducibility"
            )

        # Evaluation methodology
        if "leave-one-out" in lines and "justification" not in lines:
            feedback["suggestions"].append(
                "Justify choice of Leave-One-Out CV over other methods more explicitly"
            )

        # Architecture design choices
        if "temporal convolutional" in lines:
            feedback["suggestions"].append(
                "Explain why TCN was chosen over other sequence models (LSTM, Transformer)"
            )

        # Feature engineering justification
        if "delta" in lines and "speech recognition" in lines:
            feedback["strengths"].append(
                "Good connection to established speech recognition practices"
            )

        # Ethical considerations
        feedback["missing_elements"].append(
            "Ethics statement missing - data collection and participant consent should be addressed"
        )

        # Data preprocessing details
        if "normalization" in lines:
            feedback["strengths"].append("Appropriate audio preprocessing pipeline described")

        return feedback

    def review_results_section(self, content: str) -> Dict[str, List[str]]:
        """Critically review the Results section."""

        feedback = {
            "strengths": [],
            "critical_issues": [],
            "suggestions": [],
            "missing_elements": [],
        }

        lines = content.lower()

        # Statistical reporting
        if "p <" in content and "cohen" in lines:
            feedback["strengths"].append(
                "Appropriate statistical reporting with p-values and effect sizes"
            )

        if "confidence interval" in lines:
            feedback["strengths"].append("Inclusion of confidence intervals for robust inference")

        # Results presentation
        if "table" in lines and "figure" in lines:
            feedback["strengths"].append("Multi-modal results presentation with tables and figures")
        else:
            feedback["critical_issues"].append(
                "Results should include both tables and figures for comprehensive presentation"
            )

        # Ablation study reporting
        if "ablation" in lines:
            feedback["strengths"].append(
                "Systematic ablation study provides clear component attribution"
            )

        # Statistical significance interpretation
        if "significant" in lines and "practical" not in lines:
            feedback["suggestions"].append(
                "Distinguish between statistical and practical significance more clearly"
            )

        # Multiple comparisons
        if "bonferroni" not in lines and "correction" not in lines and "multiple" in lines:
            feedback["critical_issues"].append(
                "Multiple comparisons may require correction for inflated Type I error"
            )

        # Effect size interpretation
        if "cohen" in lines:
            feedback["suggestions"].append(
                "Provide interpretation of Cohen's d values for non-statistical readers"
            )

        # Baseline comparisons
        feedback["missing_elements"].append(
            "Comparison with published baselines or state-of-the-art methods"
        )

        return feedback

    def review_discussion_section(self, content: str) -> Dict[str, List[str]]:
        """Critically review the Discussion section."""

        feedback = {
            "strengths": [],
            "critical_issues": [],
            "suggestions": [],
            "missing_elements": [],
        }

        lines = content.lower()

        # Limitation acknowledgment
        if "limitation" in lines:
            feedback["strengths"].append("Acknowledges study limitations appropriately")
        else:
            feedback["critical_issues"].append(
                "No explicit discussion of study limitations - this is essential"
            )

        # Generalizability discussion
        if "generali" in lines:
            feedback["strengths"].append("Addresses generalizability of findings")

        # Literature integration
        if "previous" in lines or "prior" in lines:
            feedback["strengths"].append("Integrates findings with existing literature")
        else:
            feedback["critical_issues"].append("Insufficient integration with existing literature")

        # Future work
        if "future" in lines:
            feedback["strengths"].append("Provides direction for future research")

        # Clinical/practical implications
        if "application" in lines or "clinical" in lines:
            feedback["strengths"].append("Discusses practical applications of findings")
        else:
            feedback["missing_elements"].append(
                "Discussion of practical applications and implications"
            )

        # Mechanism explanation
        if "delta" in lines and "temporal" in lines:
            feedback["suggestions"].append(
                "Explain the mechanistic basis for why delta features improve performance"
            )

        return feedback

    def generate_review_report(self, section_type: str, content: str) -> str:
        """Generate a comprehensive review report."""

        if section_type == "methods":
            feedback = self.review_methods_section(content)
        elif section_type == "results":
            feedback = self.review_results_section(content)
        elif section_type == "discussion":
            feedback = self.review_discussion_section(content)
        else:
            return "Unknown section type"

        report = f"""
# Critical Review: {section_type.title()} Section

## 🟢 Strengths ({len(feedback["strengths"])})
"""
        for i, strength in enumerate(feedback["strengths"], 1):
            report += f"{i}. {strength}\n"

        report += f"\n## 🔴 Critical Issues ({len(feedback['critical_issues'])})\n"
        for i, issue in enumerate(feedback["critical_issues"], 1):
            report += f"{i}. {issue}\n"

        report += f"\n## 🟡 Suggestions for Improvement ({len(feedback['suggestions'])})\n"
        for i, suggestion in enumerate(feedback["suggestions"], 1):
            report += f"{i}. {suggestion}\n"

        report += f"\n## ⚪ Missing Elements ({len(feedback['missing_elements'])})\n"
        for i, missing in enumerate(feedback["missing_elements"], 1):
            report += f"{i}. {missing}\n"

        # Overall assessment
        total_issues = len(feedback["critical_issues"]) + len(feedback["missing_elements"])
        if total_issues == 0:
            assessment = "EXCELLENT - Publication ready with minor revisions"
        elif total_issues <= 2:
            assessment = "GOOD - Needs minor revisions"
        elif total_issues <= 5:
            assessment = "FAIR - Needs moderate revisions"
        else:
            assessment = "POOR - Needs major revisions"

        report += f"\n## Overall Assessment: {assessment}\n"

        return report


def main():
    """Main function for command-line usage."""
    if len(sys.argv) != 3:
        print("Usage: python critical_reviewer_agent.py <section_type> <file_path>")
        print("Section types: methods, results, discussion")
        sys.exit(1)

    section_type = sys.argv[1].lower()
    file_path = Path(sys.argv[2])

    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    content = file_path.read_text()
    reviewer = CriticalReviewer()
    report = reviewer.generate_review_report(section_type, content)

    print(report)

    # Save review to file
    review_file = file_path.parent / f"{file_path.stem}_review.md"
    review_file.write_text(report)
    print(f"\nReview saved to: {review_file}")


if __name__ == "__main__":
    main()
