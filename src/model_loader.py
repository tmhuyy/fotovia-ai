import torch
import torchvision.models as models
from torchvision import transforms
from .config import MODEL_NAME, NUM_CLASSES, IMG_SIZE, WEIGHTS_PATH

print(f"Loading AI model ({MODEL_NAME}) once...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    if MODEL_NAME == "resnext":
        model = models.resnext50_32x4d(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
        return model

    elif MODEL_NAME == "wide_resnet":
        model = models.wide_resnet50_2(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
        return model

    elif MODEL_NAME in ("efficientnet", "efficientnet_b0"):
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
        return model

    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")


# Initialize model architecture
model = build_model()

# Load file weights
try:
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"Weights loaded successfully from {WEIGHTS_PATH}")
except Exception as e:
    print(
        f"Warning: Could not load weights. Make sure {WEIGHTS_PATH} exists. Error: {e}"
    )

model.to(device)
model.eval()

# Transforms
img_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

print(f"Model loaded 🚀 on {device}")
