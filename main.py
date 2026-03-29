"""
Main entry point for Drift Detection Project

Currently runs:
- Gas dataset pipeline (Modules 1–3)

Future:
- Add support for intel, nasa, swat datasets
- Add CLI arguments for dataset selection
"""

from experiments.gas.run_gas_pipeline import main as run_gas_pipeline


def main():
    print("=" * 60)
    print("DRIFT DETECTION PROJECT")
    print("=" * 60)

    print("\nRunning Gas Dataset Pipeline...\n")

    run_gas_pipeline()

    print("\nPipeline execution completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()