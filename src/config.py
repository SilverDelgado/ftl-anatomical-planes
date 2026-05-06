from pathlib import Path

# --- Rutas ---
P6_DIR      = Path(__file__).parent.parent
PROYECTO_DIR = P6_DIR.parent
DATA_ROOT   = P6_DIR                               # image_path en CSV es relativa a aquí (data/ está en p6/)
CSVS_DIR    = PROYECTO_DIR / "federated_csvs" / "06_hfl_cls_plane_comparison_flair"
RESULTS_DIR = P6_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HOSPITAL_TRAIN_CSVS = [
    CSVS_DIR / f"hospital_{k}_train.csv" for k in range(1, 5)
]
GLOBAL_TEST_CSV = CSVS_DIR / "global_test.csv"

# --- Hiperparámetros (Informe final §12) ---
N_CLIENTS    = 4
ROUNDS       = 50
LOCAL_EPOCHS = 5
LR           = 1e-4   # fine-tuning ResNet-18 preentrenado
BATCH_SIZE   = 32
IMG_SIZE     = 224
SEEDS        = [42, 123, 456]

# FedProx: coeficiente del término proximal
MU_FEDPROX   = 0.1

# --- Etiquetas ---
MS_CLASS      = 1   # label_id=1 → esclerosis múltiple (clase positiva)
CONTROL_CLASS = 0
