import click
import dload
import os
import pprint

RAW_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw"
)

# flake8: noqa
REAL_LIFE_DECEPTION_DETECTION_URL = "https://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip"

AVAILABLE_DATASETS = {
    "REAL_LIFE_DECEPTION_DETECTION": REAL_LIFE_DECEPTION_DETECTION_URL,
}


@click.command()
@click.argument('dataset', nargs=1)
def main(dataset):
    if dataset not in AVAILABLE_DATASETS.keys():
        print(f"Dataset {dataset} not available.")
        pprint(f"Available datasets: {list(AVAILABLE_DATASETS.keys())}")
        return

    print(f"Downloading dataset {dataset}...")
    try:
        dload.save_unzip(
            AVAILABLE_DATASETS[dataset],
            extract_path=os.path.join(RAW_DATA_DIR, dataset.lower()),
            delete_after=True
        )
        print(f"Dataset {dataset} downloaded.")
    except Exception as e:
        print(f"Error downloading dataset {dataset}: {e}")


if __name__ == "__main__":
    main()
