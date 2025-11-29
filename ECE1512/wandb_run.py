"""Run a training job with Weights & Biases tracking enabled."""
import argparse

import main as training_main



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch training with wandb logging."
            " Mirrors main.py options but turns on wandb by default."
        )
    )
    parser.add_argument(
        "--config",
        default="config/bracs_medical_ssl_config.yml",
        help="Path to the YAML config file (same as main.py).",
    )
    parser.add_argument(
        "--attn_heads",
        "--attn_head",
        dest="attn_heads",
        type=int,
        default=None,
        help="Optionally override attention heads (forwarded to main.py).",
    )
    parser.add_argument(
        "--wandb_project",
        "--project",
        dest="wandb_project",
        default="ece1512-mil",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        "--entity",
        dest="wandb_entity",
        default="ECE_1512_b",
        help="Weights & Biases entity/team name (optional).",
    )
    parser.add_argument(
        "--wandb_run_name",
        "--run_name",
        dest="wandb_run_name",
        default=None,
        help="Optional custom run name for the wandb dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Build the argument namespace expected by main.main
    training_args = argparse.Namespace(
        config=args.config,
        attn_heads=args.attn_heads,
        wandb=True,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )
    training_main.main(training_args)


if __name__ == "__main__":
    main()
