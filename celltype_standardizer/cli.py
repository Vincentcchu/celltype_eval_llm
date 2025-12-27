"""
Command-line interface for cell-type standardization and evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

from celltype_standardizer import standardize_h5ad_and_update_mapping, evaluate_h5ad
from celltype_standardizer.standardize import get_label_coverage_report


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_standardize(args):
    """Execute standardization workflow."""
    setup_logging(args.verbose)
    
    result = standardize_h5ad_and_update_mapping(
        input_h5ad=args.input,
        obs_column=args.column,
        output_h5ad=args.output,
        output_obs_column=args.output_column,
        mapping_store_path=args.mapping_store,
        l3_vocab_path=args.vocab,
        api_key=args.api_key,
        llm_model=args.model,
        skip_llm=args.skip_llm,
    )
    
    print("\n=== Standardization Complete ===")
    print(f"Processed {len(result)} cells")
    print(f"Standardized labels in column: {args.output_column}")
    if args.output:
        print(f"Saved to: {args.output}")


def cmd_evaluate(args):
    """Execute evaluation workflow."""
    setup_logging(args.verbose)
    
    results = evaluate_h5ad(
        pred_h5ad=args.pred_input,
        pred_column=args.pred_column,
        gt_h5ad=args.gt_input if args.gt_input else None,
        gt_column=args.gt_column,
        metrics_output_path=args.output,
        mapping_store_path=args.mapping_store,
        l3_vocab_path=args.vocab,
        api_key=args.api_key,
        llm_model=args.model,
        skip_llm=args.skip_llm,
    )
    
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"Total cells: {results['dataset_info']['total_cells']}")
    
    if args.output:
        print(f"\nDetailed metrics saved to: {args.output}")


def cmd_coverage(args):
    """Generate label coverage report."""
    setup_logging(args.verbose)
    
    report = get_label_coverage_report(
        input_h5ad=args.input,
        obs_column=args.column,
        mapping_store_path=args.mapping_store,
    )
    
    print("\n=== Label Coverage Report ===")
    print(f"Total unique labels: {report['total_unique_labels']}")
    print(f"Mapped labels: {report['mapped_count']}")
    print(f"Unmapped labels: {report['unmapped_count']}")
    print(f"Coverage: {report['coverage_percent']:.1f}%")
    
    if report['unmapped_labels']:
        print(f"\nUnmapped labels ({len(report['unmapped_labels'])}):")
        for label in report['unmapped_labels']:
            print(f"  - {label}")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to: {args.output}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cell-type label standardization and evaluation for L3 taxonomy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standardize labels in a dataset
  celltype-cli standardize -i data/dataset.h5ad -c cell_type -o data/dataset_l3.h5ad
  
  # Evaluate predictions against ground truth
  celltype-cli evaluate -p data/predictions.h5ad -g data/ground_truth.h5ad \\
      --pred-column predicted_type --gt-column true_type -o metrics.json
  
  # Check label coverage before standardization
  celltype-cli coverage -i data/dataset.h5ad -c cell_type
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Standardize command
    standardize_parser = subparsers.add_parser(
        'standardize',
        help='Standardize cell-type labels to L3 taxonomy'
    )
    standardize_parser.add_argument(
        '-i', '--input', required=True,
        help='Input .h5ad file path'
    )
    standardize_parser.add_argument(
        '-c', '--column', required=True,
        help='Column name in obs containing raw cell-type labels'
    )
    standardize_parser.add_argument(
        '-o', '--output',
        help='Output .h5ad file path (optional, defaults to no output file)'
    )
    standardize_parser.add_argument(
        '--output-column', default='cell_type_level3',
        help='Column name for standardized L3 labels (default: cell_type_level3)'
    )
    standardize_parser.add_argument(
        '--mapping-store',
        help='Path to mapping store JSON file (uses default if not specified)'
    )
    standardize_parser.add_argument(
        '--vocab',
        help='Path to L3 vocabulary JSON file (uses default if not specified)'
    )
    standardize_parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    standardize_parser.add_argument(
        '--model', default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    standardize_parser.add_argument(
        '--skip-llm', action='store_true',
        help='Skip LLM calls, only use existing mappings'
    )
    standardize_parser.set_defaults(func=cmd_standardize)
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate predictions against ground truth after L3 standardization'
    )
    evaluate_parser.add_argument(
        '-p', '--pred-input', required=True,
        help='Input .h5ad file with predictions'
    )
    evaluate_parser.add_argument(
        '--pred-column', required=True,
        help='Column name in pred obs containing predicted labels'
    )
    evaluate_parser.add_argument(
        '-g', '--gt-input',
        help='Ground truth .h5ad file (if separate from predictions)'
    )
    evaluate_parser.add_argument(
        '--gt-column', required=True,
        help='Column name containing ground truth labels'
    )
    evaluate_parser.add_argument(
        '-o', '--output',
        help='Output path for metrics JSON report (optional)'
    )
    evaluate_parser.add_argument(
        '--mapping-store',
        help='Path to mapping store JSON file (uses default if not specified)'
    )
    evaluate_parser.add_argument(
        '--vocab',
        help='Path to L3 vocabulary JSON file (uses default if not specified)'
    )
    evaluate_parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    evaluate_parser.add_argument(
        '--model', default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    evaluate_parser.add_argument(
        '--skip-llm', action='store_true',
        help='Skip LLM calls, only use existing mappings'
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)
    
    # Coverage command
    coverage_parser = subparsers.add_parser(
        'coverage',
        help='Generate label coverage report'
    )
    coverage_parser.add_argument(
        '-i', '--input', required=True,
        help='Input .h5ad file path'
    )
    coverage_parser.add_argument(
        '-c', '--column', required=True,
        help='Column name in obs containing cell-type labels'
    )
    coverage_parser.add_argument(
        '-o', '--output',
        help='Output path for JSON report (optional)'
    )
    coverage_parser.add_argument(
        '--mapping-store',
        help='Path to mapping store JSON file (uses default if not specified)'
    )
    coverage_parser.set_defaults(func=cmd_coverage)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
