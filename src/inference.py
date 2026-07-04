import torch
from src.model import DeepFakeDetector
from src.pipeline import process_video

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.10

# Load model
model = DeepFakeDetector().to(DEVICE)

model.load_state_dict(
    torch.load("weights/best_model.pt", map_location=DEVICE)
)

model.eval()


def predict_video(video_path):
    video_tensor = process_video(video_path)

    video_tensor = video_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(video_tensor)

        prob = torch.sigmoid(logits).item()

    result = {
        "probability": round(prob, 4),
        "verdict": "FAKE" if prob > THRESHOLD else "REAL"
    }

    return result