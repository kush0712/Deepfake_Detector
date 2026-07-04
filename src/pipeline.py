import cv2
import torch
import numpy as np
import torchvision.transforms as T
from facenet_pytorch import MTCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAMES_PER_VIDEO = 15
FACE_SIZE = 224

# Same normalization used during training
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Load MTCNN
mtcnn = MTCNN(
    image_size=FACE_SIZE,
    margin=20,
    keep_all=False,
    post_process=False,
    device=DEVICE
)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        raise Exception("Empty video")

    # Uniform sampling
    indices = np.linspace(
        0,
        total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int
    )

    face_list = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = mtcnn(frame_rgb)

        if face is not None:
            face_np = face.permute(1, 2, 0).cpu().numpy()
            face_np = (face_np * 255).astype(np.uint8)

            face_tensor = transform(face_np)

            face_list.append(face_tensor)

    cap.release()

    if len(face_list) == 0:
        raise Exception("No faces detected")

    # Padding if fewer than 15
    while len(face_list) < FRAMES_PER_VIDEO:
        face_list.append(face_list[-1])

    video_tensor = torch.stack(face_list[:FRAMES_PER_VIDEO])

    return video_tensor