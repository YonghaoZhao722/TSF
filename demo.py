from train_source_forest import train_source_forest
from target_forest_finetune import fine_tune_target_forest
from dp_based import train_dp_based_forest


def main():
    print("Training source forest...")
    source_model_path = train_source_forest()
    print(f"Source forest saved at: {source_model_path}")

    print("\nFine-tuning target forest...")
    fine_tune_target_forest()

    print("\nTraining DP-based target forest...")
    train_dp_based_forest()

    print("\nTSF training and fine-tuning completed.")


if __name__ == "__main__":
    main()
