import click
import dload
import os
from pprint import pprint

RAW_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw"
)

# flake8: noqa
SILESIAN_DECEPTION = (
    "https://dl.acm.org/doi/10.1145/2823465.2823469"
)
MIAMI_DECEPTION = (
    "https://sc.lib.miamioh.edu/handle/2374.MIA/6067"
)
REAL_LIFE_DECEPTION_DETECTION_URL = (
    "https://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip"
)
RAVDESS = (
    "https://www.kaggle.com/api/v1/datasets/download/adrivg/ravdess-emotional-speech-video"
)

AVAILABLE_DATASETS = {
    "SILESIAN_DECEPTION": SILESIAN_DECEPTION,
    "MIAMI_DECEPTION": MIAMI_DECEPTION,
    "REAL_LIFE_DECEPTION_DETECTION": REAL_LIFE_DECEPTION_DETECTION_URL,
    "RAVDESS": RAVDESS,
}

COPYRIGHTED = ["SILESIAN_DECEPTION", "MIAMI_DECEPTION"]


@click.command()
@click.argument("dataset", nargs=1)
def main(dataset):
    if dataset not in AVAILABLE_DATASETS.keys():
        print(f"Dataset {dataset} not available.")
        pprint(f"Available datasets: {list(AVAILABLE_DATASETS.keys())}")
        return
    
    if dataset in COPYRIGHTED:
        print(f"Dataset {dataset} might be copyrighted and has to be downloaded manually at {AVAILABLE_DATASETS[dataset]}.")
        return
    
    print(f"Downloading dataset {dataset}...")
    try:
        dload.save_unzip(
            AVAILABLE_DATASETS[dataset],
            extract_path=os.path.join(RAW_DATA_DIR, dataset.lower()),
            delete_after=True,
        )
        print(f"Dataset {dataset} downloaded.")
    except Exception as e:
        print(f"Error downloading dataset {dataset}: {e}")


if __name__ == "__main__":
    main()
