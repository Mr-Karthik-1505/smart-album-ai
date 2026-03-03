# Smart Album AI 🚀

AI-powered photo sorting system using DeepFace and FastAPI.

## Features

- Face detection (RetinaFace)
- User face recognition (ArcFace embeddings)
- Cosine similarity comparison
- Automatic album sorting:
  - user
  - single
  - group
  - no_face
- Multi-image upload
- FastAPI backend

## Tech Stack

- Python 3.10
- FastAPI
- DeepFace
- RetinaFace
- ArcFace
- NumPy

## How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --reload