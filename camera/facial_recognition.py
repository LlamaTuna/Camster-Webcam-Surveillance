"""
Facial Recognition module using PyTorch with FaceNet and MTCNN.

This module provides face detection and recognition capabilities using:
- MTCNN (Multi-task Cascaded Convolutional Networks) for face detection and alignment
- InceptionResnetV1 (FaceNet) for generating 512-dimensional face embeddings
"""
import cv2
import numpy as np
import time
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import os
from datetime import datetime
from django.conf import settings
from .models import Face, RecognitionSettings
from .device_utils import get_device

import logging

logger = logging.getLogger(__name__)


class FacialRecognition:
    """
    A class used to perform facial recognition tasks, including face detection,
    feature extraction, and face matching against known faces.

    Attributes:
        device (torch.device): The compute device (CUDA or CPU).
        mtcnn (MTCNN): The face detector for detecting and aligning faces.
        resnet (InceptionResnetV1): The FaceNet model for generating face embeddings.
        known_faces_features (list): List of embeddings for known faces.
        known_faces_labels (list): List of labels corresponding to the known faces.
    """

    def __init__(self, device=None):
        """
        Initializes the FacialRecognition class, setting up the face detector,
        feature extractor, and loading known faces and their features.

        Args:
            device: Optional torch.device. If None, auto-detects CUDA/CPU.
        """
        # Set up device (CUDA if available, else CPU)
        self.device = device if device is not None else get_device()
        logger.info(f"FacialRecognition using device: {self.device}")

        # Initialize MTCNN for face detection and alignment
        # keep_all=True returns all detected faces, not just the largest one
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds for face detection stages
            factor=0.709,
            post_process=True,
            keep_all=True,
            device=self.device
        )

        # Initialize FaceNet (InceptionResnetV1) for face embeddings
        # pretrained='vggface2' provides good general face recognition
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Storage for known faces
        self.known_faces_features = []
        self.known_faces_labels = []

        # Rate-limiting and deduplication for face saves
        # {label: (last_save_timestamp, last_embedding)}
        self._recent_saves = {}
        self._save_interval = 30       # min seconds between saves for same face
        self._dedup_threshold = 0.85   # skip save if similarity > this

        # Load known faces from disk
        self.load_known_faces()

    def _preprocess_image(self, img):
        """
        Preprocesses an image for face detection.

        Args:
            img (ndarray): The input BGR image from OpenCV.

        Returns:
            ndarray: RGB image suitable for MTCNN, or None if input is invalid.
        """
        if img is None or img.size == 0:
            return None
        # Convert BGR (OpenCV) to RGB (PyTorch/PIL convention)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb_img

    def _extract_features(self, face_tensor):
        """
        Extracts 512-dimensional face embeddings from a face tensor.

        Args:
            face_tensor (torch.Tensor): Preprocessed face tensor from MTCNN.

        Returns:
            ndarray: The 512-dimensional face embedding, or None if input is invalid.
        """
        if face_tensor is None:
            return None

        with torch.no_grad():
            # Ensure tensor is on the correct device
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor.to(self.device)
            
            # Get embedding from FaceNet
            embedding = self.resnet(face_tensor)
            return embedding.cpu().numpy().flatten()

    def _detect_faces(self, img, confidence_threshold=0.90):
        """
        Detects faces in an image using MTCNN.

        Args:
            img (ndarray): The input BGR image.
            confidence_threshold (float): Minimum confidence to consider a detection valid.

        Returns:
            tuple: (boxes, probs, face_tensors) where:
                - boxes: List of bounding boxes [x1, y1, x2, y2]
                - probs: List of detection probabilities
                - face_tensors: Preprocessed face tensors ready for embedding extraction
        """
        if img is None or img.size == 0:
            logger.warning("Empty or None image provided for face detection")
            return [], [], []

        try:
            # Convert to RGB for MTCNN
            rgb_img = self._preprocess_image(img)
            if rgb_img is None:
                return [], [], []

            # Detect faces - returns aligned face tensors and bounding boxes
            boxes, probs = self.mtcnn.detect(rgb_img)
            
            if boxes is None:
                return [], [], []

            # Filter by confidence threshold
            valid_indices = [i for i, prob in enumerate(probs) if prob >= confidence_threshold]
            
            if not valid_indices:
                return [], [], []

            filtered_boxes = boxes[valid_indices]
            filtered_probs = probs[valid_indices]

            # Extract aligned face tensors for the valid detections
            face_tensors = self.mtcnn(rgb_img)
            
            if face_tensors is None:
                return [], [], []

            # Handle single face case
            if face_tensors.dim() == 3:
                face_tensors = face_tensors.unsqueeze(0)

            # Filter tensors to match filtered boxes
            if len(face_tensors) > len(valid_indices):
                face_tensors = face_tensors[valid_indices]

            return filtered_boxes.tolist(), filtered_probs.tolist(), face_tensors

        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return [], [], []

    def load_known_faces(self):
        """
        Loads known faces from the database (tagged faces) and their embeddings.
        Falls back to extracting embeddings from images if not stored.
        """
        self.known_faces_features = []
        self.known_faces_labels = []
        
        try:
            # Load tagged faces from database
            tagged_faces = Face.objects.filter(tagged=True)
            
            for face in tagged_faces:
                try:
                    # Check if embedding is stored in database
                    if face.embedding:
                        # Load pre-computed embedding
                        features = np.frombuffer(face.embedding, dtype=np.float32)
                        if len(features) == 512:  # Validate embedding size
                            self.known_faces_features.append(features)
                            self.known_faces_labels.append(face.name)
                            logger.info(f"Loaded known face from DB: {face.name}")
                            continue
                    
                    # Fall back to extracting features from image
                    if face.image and os.path.exists(face.image.path):
                        img = cv2.imread(face.image.path)
                        if img is not None:
                            boxes, probs, face_tensors = self._detect_faces(img)
                            if len(face_tensors) > 0:
                                features = self._extract_features(face_tensors[0])
                                if features is not None:
                                    self.known_faces_features.append(features)
                                    self.known_faces_labels.append(face.name)
                                    
                                    # Store embedding in database for faster loading next time
                                    face.embedding = features.astype(np.float32).tobytes()
                                    face.save(update_fields=['embedding'])
                                    logger.info(f"Loaded and cached known face: {face.name}")
                except Exception as e:
                    logger.error(f"Error loading face {face.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading known faces from database: {e}")
        
        # Also load from known_faces directory for backward compatibility
        known_faces_dir = settings.KNOWN_FACES_DIR
        if os.path.exists(known_faces_dir):
            for filename in os.listdir(known_faces_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Check if this file is already loaded via database
                    label = os.path.splitext(filename)[0]
                    if any(label in l for l in self.known_faces_labels):
                        continue
                        
                    img_path = os.path.join(known_faces_dir, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            boxes, probs, face_tensors = self._detect_faces(img)
                            if len(face_tensors) > 0:
                                features = self._extract_features(face_tensors[0])
                                if features is not None:
                                    self.known_faces_features.append(features)
                                    self.known_faces_labels.append(label)
                                    logger.info(f"Loaded known face from file: {label}")
                    except Exception as e:
                        logger.error(f"Error loading face from file {label}: {e}")

        logger.info(f"Loaded {len(self.known_faces_labels)} known faces total")

    def recognize_faces(self, frame, recognition_threshold=None):
        """
        Recognizes faces in a given frame by comparing them to known faces.

        Args:
            frame (ndarray): The input BGR frame to recognize faces in.
            recognition_threshold (float): Cosine similarity threshold for recognition.
                If None, loads from database settings. Higher values require closer matches (0.0-1.0).

        Returns:
            list: A list of recognized faces with labels and coordinates.
                Each face dict contains: 'box' [x, y, w, h], 'label', 'confidence'
        """
        # Load threshold from database if not provided
        if recognition_threshold is None:
            try:
                settings_obj = RecognitionSettings.get_settings()
                recognition_threshold = settings_obj.similarity_threshold
            except Exception:
                recognition_threshold = 0.6  # Default fallback
        
        boxes, probs, face_tensors = self._detect_faces(frame)
        recognized_faces = []

        if len(boxes) == 0:
            return recognized_faces

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            try:
                # Convert from [x1, y1, x2, y2] to [x, y, w, h] format
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x, y = x1, y1
                width, height = x2 - x1, y2 - y1

                # Validate coordinates
                if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
                    continue

                # Extract features for this face
                if i < len(face_tensors):
                    features = self._extract_features(face_tensors[i])
                else:
                    continue

                if features is None:
                    continue

                # Compare against known faces using cosine similarity
                label = 'Unknown'
                best_similarity = 0.0

                for known_features, known_label in zip(self.known_faces_features, self.known_faces_labels):
                    # Cosine similarity (1.0 = identical, 0.0 = orthogonal)
                    similarity = 1 - cosine(features, known_features)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        if similarity >= recognition_threshold:
                            label = known_label

                face_data = {
                    'box': [x, y, width, height],
                    'label': label,
                    'confidence': prob,
                    'similarity': best_similarity
                }
                recognized_faces.append(face_data)

                # Save face image (with rate-limiting and dedup)
                face_img = frame[y:y + height, x:x + width]
                if face_img.size > 0:
                    self.save_face_image(face_img, label, embedding=features)

            except Exception as e:
                logger.error(f"Error recognizing face: {e}")
                continue

        return recognized_faces

    def save_face_image(self, face_img, label, embedding=None):
        """
        Saves the recognized face image to disk and creates a corresponding record in the database.
        Rate-limited to one save per face identity per save_interval seconds.
        Deduplicated by embedding similarity to avoid saving near-identical captures.

        Args:
            face_img (ndarray): The face image to save.
            label (str): The label of the face (e.g., name of the person).
            embedding (ndarray, optional): The 512-dim face embedding for dedup comparison.
        """
        try:
            now = time.time()

            # Rate-limit check: skip if same face was saved recently
            if label in self._recent_saves:
                last_time, last_emb = self._recent_saves[label]
                if now - last_time < self._save_interval:
                    return  # Too soon since last save of this face
                # Dedup check: skip if embedding is too similar to last save
                if embedding is not None and last_emb is not None:
                    similarity = 1 - cosine(embedding, last_emb)
                    if similarity > self._dedup_threshold:
                        return  # Too similar to last saved capture

            faces_seen_dir = os.path.join(settings.MEDIA_ROOT, 'faces_seen')
            if not os.path.exists(faces_seen_dir):
                os.makedirs(faces_seen_dir)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{label}_{timestamp}.jpg"
            filepath = os.path.join(faces_seen_dir, filename)

            cv2.imwrite(filepath, face_img)
            logger.debug(f"Face image saved: {filepath}")

            Face.objects.create(name=label, image=f"faces_seen/{filename}")
            logger.debug(f"Face record saved: {label}, {filename}")

            # Update recent saves tracker
            self._recent_saves[label] = (now, embedding)

        except Exception as e:
            logger.error(f"Error saving face image: {e}")

    def add_known_face(self, img, label):
        """
        Adds a new known face to the recognition database.

        Args:
            img (ndarray): The image containing the face.
            label (str): The label/name for this face.

        Returns:
            bool: True if face was successfully added, False otherwise.
        """
        boxes, probs, face_tensors = self._detect_faces(img)
        
        if len(face_tensors) == 0:
            logger.warning(f"No face detected when adding known face: {label}")
            return False

        features = self._extract_features(face_tensors[0])
        if features is not None:
            self.known_faces_features.append(features)
            self.known_faces_labels.append(label)
            logger.info(f"Added known face: {label}")
            return True

        return False

    def reload_known_faces(self):
        """
        Reloads all known faces from disk, clearing existing cache.
        """
        self.known_faces_features = []
        self.known_faces_labels = []
        self.load_known_faces()
