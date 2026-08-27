from __future__ import annotations
import argparse
import sys
from app.replay.runner import replay_experiment

def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an SIH26147 experiment bundle deterministically")
    parser.add_argument("experiment_file", type=str, help="Path to experiment.json")
    args = parser.parse_args()

    print(f"Replaying experiment from: {args.experiment_file}...")
    res, matches = replay_experiment(args.experiment_file)
    print(f"Reproducibility Hash Match: {matches}")
    print(f"Final Assessment: {res.final_assessment_text}")
    print(f"Verified: {res.is_verified}")

if __name__ == "__main__":
    main()
