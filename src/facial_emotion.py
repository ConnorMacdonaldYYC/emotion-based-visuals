import threading
import time

import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from cv2.typing import MatLike
from transformers import ViTForImageClassification, ViTImageProcessor


class EmotionAnalyzer:
    def __init__(self, img_size=224, device="cpu"):
        self.img_size = img_size
        self.device = torch.device(device)
        model, feature_extractor = self._load_model()
        self.model = model
        self.feature_extractor = feature_extractor
        self.face_detector = dlib.get_frontal_face_detector()
        self.emotions = {}  # {(x, y, w, h): emotion_label}
        self.frame = None
        self.analyzing = False

    def _load_model(self) -> tuple[ViTForImageClassification, ViTImageProcessor]:
        feature_extractor = ViTImageProcessor.from_pretrained(
            "trpakov/vit-face-expression"
        )
        model = ViTForImageClassification.from_pretrained("trpakov/vit-face-expression")

        return model, feature_extractor

    def preprocess_frame(self, frame: MatLike) -> torch.Tensor:
        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((self.img_size, self.img_size)),
                # transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )
        return preprocess(frame)

    def map_index_to_emotion(self, index: int) -> str:
        return self.model.config.id2label[index]

    def predict_emotion(self, face: MatLike) -> int:
        frame_tensor = self.preprocess_frame(face)
        with torch.no_grad():
            inputs = self.feature_extractor(images=frame_tensor, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

            return predicted_class_idx

    def analyze_frame(self, frame: MatLike, faces: list):
        self.analyzing = True
        emotions = {}
        for x, y, w, h in faces:
            face_image = frame[y : y + h, x : x + w]
            emotion_index = self.predict_emotion(face_image)
            emotion_label = self.map_index_to_emotion(emotion_index)
            emotions[(x, y, w, h)] = emotion_label
        self.emotions = emotions
        self.analyzing = False

    def draw_emotions(self, frame: MatLike, faces: list):
        for x, y, w, h in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2,
            )
            if self.emotions:
                closest_face = min(
                    self.emotions.keys(), key=lambda f: self.distance(f, (x, y, w, h))
                )
                emotion_label = self.emotions[closest_face]
                cv2.putText(
                    frame,
                    emotion_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

    def distance(self, face1, face2):
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)


def analyze_emotions(analyzer, frame, faces, delay=1):
    analyzer.analyze_frame(frame, faces)


if __name__ == "__main__":
    device = "mps"
    analyzer = EmotionAnalyzer(device=device)
    cap = cv2.VideoCapture(0)  # Open the default camera
    while True:
        ret, frame = cap.read()
        faces = analyzer.face_detector(frame, 1)
        face_coords = [
            (face.left(), face.top(), face.width(), face.height()) for face in faces
        ]
        if not analyzer.analyzing:
            threading.Thread(
                target=analyze_emotions, args=(analyzer, frame, face_coords)
            ).start()
        analyzer.draw_emotions(frame, face_coords)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
