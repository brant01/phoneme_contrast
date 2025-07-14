#!/usr/bin/env python3
"""
Systematic implementation of critical reviewer concerns.
Execute each phase step-by-step to address all methodological issues.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class ReviewerConcern:
    """Represents a single reviewer concern with implementation details."""

    id: str
    priority: str
    issue: str
    commands: List[str]
    estimated_time: str
    success_criteria: List[str]
    status: str = "pending"


class ReviewerChecklistManager:
    """Manages systematic implementation of reviewer concerns."""

    def __init__(self, config_path: str = "configs/critical_reviewer.yaml"):
        self.config_path = Path(config_path)
        self.concerns = self._load_concerns()
        self.results_log = Path("reviewer_implementation_log.json")

    def _load_concerns(self) -> Dict[str, ReviewerConcern]:
        """Load concerns from YAML configuration."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        concerns = {}

        # High priority concerns
        for concern_id, data in config["reviewer_checklist"]["high_priority"].items():
            concerns[concern_id] = ReviewerConcern(
                id=concern_id,
                priority="high",
                issue=data["issue"],
                commands=self._extract_commands(concern_id, config),
                estimated_time=data["estimated_time"],
                success_criteria=self._extract_success_criteria(concern_id, config),
            )

        # Medium priority concerns
        for concern_id, data in config["reviewer_checklist"]["medium_priority"].items():
            concerns[concern_id] = ReviewerConcern(
                id=concern_id,
                priority="medium",
                issue=data["issue"],
                commands=self._extract_commands(concern_id, config),
                estimated_time=data["estimated_time"],
                success_criteria=self._extract_success_criteria(concern_id, config),
            )

        return concerns

    def _extract_commands(self, concern_id: str, config: Dict) -> List[str]:
        """Extract implementation commands for a concern."""
        commands = []

        # Look in quick_commands section
        quick_commands = config.get("quick_commands", {})

        if concern_id == "missing_ablation_studies":
            commands.extend(
                [
                    quick_commands.get("ablation_studies", {}).get("just_more_mfcc", ""),
                    quick_commands.get("ablation_studies", {}).get("just_delta", ""),
                    quick_commands.get("ablation_studies", {}).get("just_delta_delta", ""),
                ]
            )
        elif concern_id == "statistical_significance_testing":
            commands.append(quick_commands.get("multiple_seeds", {}).get("seed_runs", ""))
        elif concern_id == "missing_baseline_comparisons":
            commands.append(quick_commands.get("baseline_comparison", {}).get("svm_baseline", ""))

        return [cmd for cmd in commands if cmd.strip()]

    def _extract_success_criteria(self, concern_id: str, config: Dict) -> List[str]:
        """Extract success criteria for a concern."""
        criteria_map = config.get("success_criteria", {})

        if concern_id == "missing_ablation_studies":
            return criteria_map.get("ablation_study", [])
        elif concern_id == "statistical_significance_testing":
            return criteria_map.get("statistical_significance", [])
        elif concern_id == "missing_baseline_comparisons":
            return criteria_map.get("baseline_comparison", [])

        return []

    def display_phase_plan(self, phase: str = "phase_1_immediate"):
        """Display implementation plan for a specific phase."""
        print(f"\n🎯 REVIEWER CONCERN IMPLEMENTATION: {phase.upper()}")
        print("=" * 70)

        # Get phase tasks from config
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        phase_data = config["implementation_plan"][phase]
        print(f"Priority: {phase_data['priority']}")
        print(f"Tasks: {', '.join(phase_data['tasks'])}")

        # Show relevant concerns
        phase_concerns = []
        for task in phase_data["tasks"]:
            if "statistical" in task:
                phase_concerns.append("statistical_significance_testing")
            elif "evaluation" in task:
                phase_concerns.append("evaluation_methodology_inconsistencies")
            elif "ablation" in task:
                phase_concerns.append("missing_ablation_studies")
            elif "baseline" in task:
                phase_concerns.append("missing_baseline_comparisons")

        print("\n📋 CONCERNS TO ADDRESS:")
        print("-" * 40)

        total_time = 0
        for concern_id in phase_concerns:
            if concern_id in self.concerns:
                concern = self.concerns[concern_id]
                print(f"\n🔍 {concern.id.replace('_', ' ').title()}")
                print(f"   Issue: {concern.issue}")
                print(f"   Priority: {concern.priority}")
                print(f"   Time: {concern.estimated_time}")
                print(f"   Status: {concern.status}")

                if concern.commands:
                    print("   Commands ready: ✅")
                else:
                    print("   Commands ready: ❌")

    def execute_concern(self, concern_id: str, dry_run: bool = True):
        """Execute implementation for a specific concern."""
        if concern_id not in self.concerns:
            print(f"❌ Unknown concern: {concern_id}")
            return False

        concern = self.concerns[concern_id]
        print(f"\n🚀 IMPLEMENTING: {concern.id.replace('_', ' ').title()}")
        print(f"Issue: {concern.issue}")
        print(f"Priority: {concern.priority}")
        print(f"Estimated time: {concern.estimated_time}")

        if not concern.commands:
            print("❌ No implementation commands available")
            return False

        print("\n📝 IMPLEMENTATION COMMANDS:")
        for i, cmd in enumerate(concern.commands, 1):
            print(f"{i}. {cmd}")

        if dry_run:
            print("\n⚠️  DRY RUN MODE - Commands not executed")
            print(f"To execute, run: python scripts/reviewer_checklist.py --execute {concern_id}")
            return True

        # Execute commands
        results = []
        for i, cmd in enumerate(concern.commands, 1):
            print(f"\n⏳ Executing command {i}/{len(concern.commands)}...")
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                results.append(
                    {
                        "command": cmd,
                        "success": result.returncode == 0,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )

                if result.returncode == 0:
                    print(f"✅ Command {i} completed successfully")
                else:
                    print(f"❌ Command {i} failed: {result.stderr}")

            except Exception as e:
                print(f"❌ Command {i} error: {e}")
                results.append({"command": cmd, "success": False, "error": str(e)})

        # Log results
        self._log_results(concern_id, results)

        # Update status
        if all(r["success"] for r in results):
            concern.status = "completed"
            print(f"\n🎉 CONCERN FULLY IMPLEMENTED: {concern_id}")
        else:
            concern.status = "failed"
            print(f"\n😞 CONCERN IMPLEMENTATION FAILED: {concern_id}")

        return concern.status == "completed"

    def _log_results(self, concern_id: str, results: List[Dict]):
        """Log implementation results."""
        log_data = {}
        if self.results_log.exists():
            with open(self.results_log) as f:
                log_data = json.load(f)

        log_data[concern_id] = {"timestamp": str(Path().cwd()), "results": results}

        with open(self.results_log, "w") as f:
            json.dump(log_data, f, indent=2)

    def show_progress(self):
        """Show overall progress on reviewer concerns."""
        print("\n📊 REVIEWER CONCERNS PROGRESS")
        print("=" * 50)

        by_priority = {"high": [], "medium": [], "low": []}
        for concern in self.concerns.values():
            by_priority[concern.priority].append(concern)

        for priority in ["high", "medium"]:
            if not by_priority[priority]:
                continue

            print(f"\n🔥 {priority.upper()} PRIORITY")
            print("-" * 30)

            for concern in by_priority[priority]:
                status_emoji = {
                    "pending": "⏳",
                    "in_progress": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                }.get(concern.status, "❓")

                print(f"{status_emoji} {concern.id.replace('_', ' ').title()}")
                print(f"   {concern.issue[:60]}...")

        # Overall stats
        total = len(self.concerns)
        completed = sum(1 for c in self.concerns.values() if c.status == "completed")
        pending = sum(1 for c in self.concerns.values() if c.status == "pending")

        print(f"\n📈 OVERALL PROGRESS: {completed}/{total} completed ({pending} pending)")


def main():
    """Main implementation interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Reviewer concern implementation")
    parser.add_argument(
        "--phase", default="phase_1_immediate", help="Implementation phase to display"
    )
    parser.add_argument("--execute", help="Execute specific concern")
    parser.add_argument("--progress", action="store_true", help="Show progress overview")
    parser.add_argument(
        "--dry-run", action="store_true", default=True, help="Show commands without executing"
    )

    args = parser.parse_args()

    manager = ReviewerChecklistManager()

    if args.progress:
        manager.show_progress()
    elif args.execute:
        manager.execute_concern(args.execute, dry_run=args.dry_run)
    else:
        manager.display_phase_plan(args.phase)


if __name__ == "__main__":
    main()
