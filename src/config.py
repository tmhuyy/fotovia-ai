MODEL_NAME = "resnext"
NUM_CLASSES = 10
IMG_SIZE = 224

# List of classes in dataset
LABELS = [
    "aerial",
    "architecture",
    "event",
    "fashion",
    "food",
    "nature",
    "sports",
    "street",
    "wedding",
    "wildlife"
]

WEIGHTS_PATH = f"{MODEL_NAME}_best.pth"