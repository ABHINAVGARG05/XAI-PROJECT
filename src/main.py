"""Full pipeline: preprocess -> train -> evaluate -> print report.

Usage: python -m src.main
"""

from src.evaluate import run_evaluation
from src.preprocess import load_and_split
from src.train import train_model


def main() -> None:
    splits = load_and_split()
    model = train_model(splits)
    run_evaluation(model, splits)


if __name__ == "__main__":
    main()
