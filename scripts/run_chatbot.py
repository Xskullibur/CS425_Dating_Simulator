#!/usr/bin/env python3
"""
VN Dating Simulator Chatbot Launcher

Convenience script to launch the VN dating simulator CLI.
This is a simple wrapper around src.app.cli.main().

Usage:
    python scripts/run_chatbot.py
    python scripts/run_chatbot.py --name Alice --character Monika --model checkpoints/vn_model
    python scripts/run_chatbot.py --model-type merged --character Sayori --model checkpoints/dating_sim_vn_merged/final
"""

import sys
import argparse
from src.app.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VN Dating Simulator - Interactive CLI"
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Your name (prompts if not provided)"
    )
    parser.add_argument(
        "--character", "-c",
        default=None,
        help="Character to talk with (Monika, Sayori, Natsuki, Yuri). Prompts if not provided."
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to trained model checkpoint (e.g., checkpoints/vn_model)"
    )
    parser.add_argument(
        "--model-type", "-t",
        default="individual",
        choices=["individual", "merged"],
        help="Model type: 'individual' (character-specific, default) or 'merged' (unified, character is cosmetic)"
    )

    args = parser.parse_args()
    sys.exit(main(character=args.character, model_path=args.model, model_type=args.model_type, user_name=args.name) or 0)
