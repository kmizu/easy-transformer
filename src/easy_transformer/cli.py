"""Command-line interface for easy-transformer."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import __version__


def train_command(args):
    """Handle the train command."""
    print(f"Training model with config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # TODO: Implement actual training logic
    print("Training functionality not yet implemented.")


def evaluate_command(args):
    """Handle the evaluate command."""
    print(f"Evaluating model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    
    # TODO: Implement actual evaluation logic
    print("Evaluation functionality not yet implemented.")


def generate_command(args):
    """Handle the generate command."""
    print(f"Generating text with model: {args.model_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Max length: {args.max_length}")
    print(f"Temperature: {args.temperature}")
    
    # TODO: Implement actual generation logic
    print("Generation functionality not yet implemented.")


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="easy-transformer",
        description="Easy Transformer - A comprehensive tutorial implementation",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a transformer model")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    train_parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing training data",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to save model and logs",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    eval_parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="Path to test data",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    eval_parser.set_defaults(func=evaluate_command)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text using a trained model")
    gen_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    gen_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for generation",
    )
    gen_parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum length of generated text",
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    gen_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    gen_parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    gen_parser.set_defaults(func=generate_command)
    
    return parser


def main(argv: Optional[list] = None):
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()