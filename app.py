import os
import shutil
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import List
from typing_extensions import Annotated
from deepface import DeepFace

app = FastAPI()

UPLOAD_FOLDER = "uploads"
USER_FOLDER = "user_faces"

ALBUMS = {
    "single": "albums/single",
    "group": "albums/group",
    "user": "albums/user",
    "no_face": "albums/no_face"
}

# Ensure folders exist
for folder in ALBUMS.values():
    os.makedirs(folder, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load user reference image
def get_user_image():
    images = os.listdir(USER_FOLDER)
    if images:
        return os.path.join(USER_FOLDER, images[0])
    return None

USER_IMAGE = get_user_image()

# 🔥 Compute user embedding ONCE at startup
def get_user_embedding():
    if not USER_IMAGE:
        return None

    embedding = DeepFace.represent(
        img_path=USER_IMAGE,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=False
    )

    print("User embedding loaded")
    return np.array(embedding[0]["embedding"])

USER_EMBEDDING = get_user_embedding()


# Simple HTML Upload Page
@app.get("/", response_class=HTMLResponse)
async def upload_page():
    return """
    <html>
        <body>
            <h2>Upload Multiple Images</h2>
            <form action="/upload/" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple>
                <input type="submit">
            </form>
        </body>
    </html>
    """


@app.post("/upload/")
async def upload_images(
    files: Annotated[List[UploadFile], File()]
):
    results = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        process_image(file_path)
        results.append(file.filename)

    return {
        "message": "Images processed successfully",
        "files_processed": results
    }


def cosine_distance(vec1, vec2):
    return 1 - np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


def process_image(image_path):
    try:
        print(f"\nProcessing: {image_path}")

        # Extract faces using RetinaFace
        faces_raw = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=False
        )

        # Filter weak detections
        faces = [f for f in faces_raw if f["confidence"] > 0.90]

        face_count = len(faces)
        print(f"Strong faces detected: {face_count}")

        # No face case
        if face_count == 0:
            shutil.move(image_path, os.path.join(ALBUMS["no_face"], os.path.basename(image_path)))
            print("Moved to no_face album")
            return

        # 🔥 USER MATCH USING EMBEDDINGS
        if USER_EMBEDDING is not None:
            for face in faces:
                temp_path = "temp_face.jpg"
                face_img = face["face"]

                # Save cropped face
                Image.fromarray((face_img * 255).astype("uint8")).save(temp_path)

                # Get embedding of cropped face
                embedding = DeepFace.represent(
                    img_path=temp_path,
                    model_name="ArcFace",
                    detector_backend="skip",
                    enforce_detection=False
                )

                os.remove(temp_path)

                current_embedding = np.array(embedding[0]["embedding"])

                distance = cosine_distance(USER_EMBEDDING, current_embedding)
                print("Cosine distance:", distance)

                # Threshold (tune if needed)
                if distance < 0.5:
                    shutil.move(image_path, os.path.join(ALBUMS["user"], os.path.basename(image_path)))
                    print("Moved to user album")
                    return

        # If not user → sort by face count
        if face_count == 1:
            shutil.move(image_path, os.path.join(ALBUMS["single"], os.path.basename(image_path)))
            print("Moved to single album")
        else:
            shutil.move(image_path, os.path.join(ALBUMS["group"], os.path.basename(image_path)))
            print("Moved to group album")

    except Exception as e:
        print("ERROR PROCESSING IMAGE:", e)