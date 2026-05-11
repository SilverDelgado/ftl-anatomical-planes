from pathlib import Path

P6_DIR = Path(__file__).parent.parent
PROYECTO_DIR = P6_DIR.parent
DATA_ROOT = P6_DIR
CSVS_DIR = PROYECTO_DIR / "federated_csvs" / "06_hfl_cls_plane_comparison_flair"
RESULTS_DIR = P6_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HOSPITAL_TRAIN_CSVS = []
for k in range(1, 5):
    HOSPITAL_TRAIN_CSVS.append(CSVS_DIR / f"hospital_{k}_train.csv")
GLOBAL_TEST_CSV = CSVS_DIR / "global_test.csv"

N_CLIENTS = 4
ROUNDS = 5
LOCAL_EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 32
IMG_SIZE = 224
SEEDS = [42, 123]

MU_FEDPROX = 0.1
GRAD_CLIP_NORM = 1.0

MS_CLASS = 1
CONTROL_CLASS = 0
