import argparse
import base64
import io
import time

import cv2
import numpy as np
import requests
import torch
import torch.multiprocessing
from PIL import Image

from src.audio_generation import AudioGenerator
from src.facial_emotion import EmotionAnalyzer
from src.generate_image import EmotionBasedImageGenerator

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def analyse_frame_emotions_and_display(analyzer: EmotionAnalyzer, frame) -> list[str]:
    faces = analyzer.face_detector(frame, 1)
    face_coords = [
        (face.left(), face.top(), face.width(), face.height()) for face in faces
    ]
    analyzer.analyze_frame(frame, face_coords)
    analyzer.draw_emotions(frame, face_coords)
    cv2.imshow("Emotion Detection", frame)
    cv2.waitKey(1)
    return list(analyzer.emotions.values())


def countdown(countdown_time: int, cap: cv2.VideoCapture):
    start_time = cv2.getTickCount()
    elapsed_time = 0
    while elapsed_time < countdown_time:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate remaining time
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        remaining_time = max(0, countdown_time - int(elapsed_time))

        # Display countdown
        cv2.putText(
            frame,
            str(remaining_time),
            (frame.shape[1] // 2, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
        )

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def generate_images(
    image_generator: EmotionBasedImageGenerator, emotion_list: list[str]
) -> tuple[list[Image.Image], float]:

    time_start = time.time()
    images = image_generator.generate_image_from_emotions(emotion_list)
    total_time = time.time() - time_start
    return images, total_time


def display_images(
    images: list[Image.Image],
    emotion_list: list[str],
    prev_emotion_list: list[str],
    transition_for: float,
):
    num_images = len(images)
    if num_images == 0:
        return

    display_time = transition_for / num_images

    for i, image in enumerate(images):
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Prepare text to display
        if i == len(images) - 1 or len(prev_emotion_list) == 0:
            emotion_text = ", ".join(emotion_list)
        else:
            emotion_text = (
                f"{', '.join(prev_emotion_list)} -> {', '.join(emotion_list)}"
            )

        # Put text on image
        cv2.putText(
            opencv_image,
            emotion_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Display the generated image
        cv2.imshow("Generated Image", opencv_image)
        cv2.waitKey(int(display_time * 1000))  # Convert seconds to milliseconds


def image_producer(
    queue: torch.multiprocessing.Queue,
):
    device = DEVICE
    analyzer = EmotionAnalyzer(device=device)
    image_generator = EmotionBasedImageGenerator(
        base_prompt="a detailed landscape painting designed to evoke the provided emotions",
        max_word_count=50,
        device=device,
    )
    cap = cv2.VideoCapture(0)  # Open the default camera
    while True:
        start_time = time.time()
        countdown(3, cap)

        ret, frame = cap.read()
        emotion_list = analyse_frame_emotions_and_display(
            analyzer=analyzer, frame=frame
        )
        if len(emotion_list) != 0:
            images, _ = generate_images(
                image_generator=image_generator, emotion_list=emotion_list
            )
            queue.put_nowait((emotion_list, images, time.time() - start_time))

        time.sleep(1)


def image_consumer(
    queue: torch.multiprocessing.Queue,
):
    prev_emotion_list = []
    fade_in_duration = 1000  # This is in ms, so right now it's 1 second. Both of these parameters could be changed based on refresh rate used.
    fade_out_duration = 3000
    audio_generator = AudioGenerator()

    while True:
        emotion_list, images, generate_time = queue.get()
        if len(emotion_list) > 0:
            display_images(
                images=images,
                emotion_list=emotion_list,
                prev_emotion_list=prev_emotion_list,
                transition_for=generate_time - 3,
            )

            if audio_generator.current_sound:
                audio_generator.fade_out_audio(fade_out_duration)
            most_common_emotion = max(set(emotion_list), key=emotion_list.count)
            audio_generator.play_audio(most_common_emotion, fade_in_duration)
            prev_emotion_list = emotion_list
        del emotion_list, images, generate_time
        time.sleep(1)


if __name__ == "__main__":

    queue = torch.multiprocessing.Queue(1)
    p1 = torch.multiprocessing.Process(target=image_producer, args=(queue,))
    p2 = torch.multiprocessing.Process(
        target=image_consumer,
        args=(queue,),
    )

    p1.start()
    p2.start()

    try:
        # Keep the main process running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected.")
        print("Terminating processes...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
        print("Processes terminated.")
    except Exception as e:
        print("Terminating processes...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
        print("Processes terminated.")
        raise e
