from dotenv import load_dotenv
from safety_experiment import SafetyExperiment

load_dotenv(dotenv_path="../.env")


def main():
    result = SafetyExperiment()
    result.run_experiment()


if __name__ == "__main__":
    main()
