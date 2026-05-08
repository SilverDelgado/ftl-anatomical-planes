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

# hiperparams
N_CLIENTS = 4
ROUNDS = 20
LOCAL_EPOCHS = 5
LR = 1e-4 # finetuning ResNet18 pretrained
BATCH_SIZE = 32
IMG_SIZE = 224
SEEDS = [42]

# FedProx
MU_FEDPROX   = 0.1

#labels
MS_CLASS      = 1 # label_id=1 -> esclerosis múltiple (EM)
CONTROL_CLASS = 0 # label_id=0 -> control (sin EM)
