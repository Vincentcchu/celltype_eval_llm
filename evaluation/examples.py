#!/usr/bin/env python3
"""
Example script demonstrating the batch evaluation pipeline.

This script shows various usage patterns.
"""

from pathlib import Path
import subprocess
import sys


def run_command(cmd: str, description: str) -> None:
    """Run a command and print its description."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n⚠️  Command failed with exit code {result.returncode}")
    else:
        print(f"\n✓ Command succeeded")


def main():
    """Run example commands."""
    
    print("="*60)
    print("BATCH EVALUATION PIPELINE - EXAMPLES")
    print("="*60)
    
    # Check if we're in the right directory
    script_dir = Path(__file__).parent
    if not (script_dir / 'run_evaluation.py').exists():
        print("Error: Please run this script from the evaluation/ directory")
        return 1
    
    examples = [
        (
            "python run_evaluation.py --help",
            "Show help message"
        ),
        (
            "python run_evaluation.py --tissue colorectal --skip-llm --no-plots",
            "Evaluate colorectal tissue (fast, no LLM calls, no plots)"
        ),
        (
            "python run_evaluation.py --tissue brain breast --skip-llm",
            "Evaluate multiple tissues (brain and breast)"
        ),
        (
            "python run_evaluation.py --tissue colorectal --output-dir test_results",
            "Custom output directory"
        ),
    ]
    
    print("\nAvailable example commands:")
    for i, (cmd, desc) in enumerate(examples, 1):
        print(f"\n{i}. {desc}")
        print(f"   {cmd}")
    
    print("\n" + "="*60)
    print("To run an example, use this script or copy the command above")
    print("="*60)
    
    # Ask user if they want to run an example
    choice = input("\nEnter example number to run (or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        return 0
    
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(examples):
            cmd, desc = examples[choice_idx]
            run_command(cmd, desc)
        else:
            print(f"Invalid choice. Please enter 1-{len(examples)}")
            return 1
    except ValueError:
        print("Invalid input. Please enter a number or 'q'")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
