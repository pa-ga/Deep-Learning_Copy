from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
RAW_DATA_DIR = ROOT / 'data'
PROCESSED_DATA_DIR = ROOT / 'processed_data'
SPLITS_DIR = ROOT / 'data_splits'
CHECKPOINTS_DIR = ROOT / 'checkpoints'
RESULTS_UNFILTERED_DIR = ROOT / 'results_unfiltered'
RESULTS_FILTERED_DIR = ROOT / 'results_filtered'

assert ROOT.is_dir()
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
SPLITS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)
RESULTS_UNFILTERED_DIR.mkdir(exist_ok=True)
RESULTS_FILTERED_DIR.mkdir(exist_ok=True)
