import cv2
import dlib
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
from scipy.spatial import distance
from mtcnn.mtcnn import MTCNN
import os
from datetime import datetime
from django.conf import settings
from .models import Face

class FacialRecognition:
    """
    A class used to perform facial recognition tasks, including face detection,
    feature extraction, and face matching against known faces.

    Attributes:
        detector (MTCNN): The face detector used to detect faces in images.
        base_model (ResNet50): The base ResNet50 model for feature extraction.
        model (Model): The final feature extractor model.
        known_faces_features (list): List of features for known faces.
        known_faces_labels (list): List of labels corresponding to the known faces.
        shape_predictor (dlib.shape_predictor): Dlib's shape predictor for face alignment.
    """

    def __init__(self):
        """
        Initializes the FacialRecognition class, setting up the face detector,
        feature extractor, and loading known faces and their features.
        """
        self.detector = MTCNN()
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.model = self._build_feature_extractor(self.base_model)
        self.known_faces_features = []
        self.known_faces_labels = []
        
        # Load the shape predictor
        shape_predictor_path = os.path.join(settings.MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
        if os.path.exists(shape_predictor_path):
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        else:
            self.shape_predictor = None
        
        self.load_known_faces()

    def _build_feature_extractor(self, base_model):
        """
        Builds a feature extractor model on top of the base ResNet50 model.

        Args:
            base_model (ResNet50): The base model to extend.

        Returns:
            Model: The constructed feature extractor model.
        """
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(128, activation='relu')(x)
        return Model(inputs=base_model.input, outputs=predictions)

    def _preprocess_image(self, img):
        """
        Preprocesses the image for feature extraction.

        Args:
            img (ndarray): The input image.

        Returns:
            ndarray: The preprocessed image array or None if input is invalid.
        """
        if img is None or img.size == 0:
            return None
        img = cv2.resize(img, (224, 224))
        img_array = np.array(img, dtype='float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def _extract_features(self, img_array):
        """
        Extracts features from the preprocessed image.

        Args:
            img_array (ndarray): The preprocessed image array.

        Returns:
            ndarray: The extracted features or None if input is invalid.
        """
        if img_array is None:
            return None
        features = self.model.predict(img_array)
        return features.flatten()

    def _detect_faces(self, img, confidence_threshold=0.70):
        """
        Detects faces in an image using the MTCNN detector.

        Args:
            img (ndarray): The input image.
            confidence_threshold (float): Minimum confidence to consider a detection valid.

        Returns:
            list: A list of detected faces with coordinates and confidence levels.
        """
        if img is None or img.size == 0:
            print("Error: The image provided for face detection is empty or None.")
            return []  # Return an empty list if the image is invalid

        try:
            small_img = cv2.resize(img, (160, 120))
        except cv2.error as e:
            print(f"Error resizing image: {e}")
            return []  # Return an empty list if resizing fails

        faces = self.detector.detect_faces(small_img)
        for face in faces:
            face['box'] = [int(coordinate * 2) for coordinate in face['box']]
        filtered_faces = [face for face in faces if face['confidence'] >= confidence_threshold]
        return filtered_faces

    def _align_face(self, img, box):
        """
        Aligns the face based on facial landmarks using dlib's shape predictor.

        Args:
            img (ndarray): The input image.
            box (tuple): The bounding box of the face.

        Returns:
            ndarray: The aligned face image or the original image if alignment fails.
        """
        x, y, w, h = box
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        if self.shape_predictor:
            landmarks = self.shape_predictor(img, dlib_rect)
            points = [(p.x, p.y) for p in landmarks.parts()]
            aligned_face = self._warp_affine(img, points)
            return aligned_face
        else:
            print("Shape predictor is not loaded.")
            return img

    def _warp_affine(self, img, points):
        """
        Applies affine transformation to the face for alignment.

        Args:
            img (ndarray): The input image.
            points (list): List of landmark points for affine transformation.

        Returns:
            ndarray: The warped (aligned) face image.
        """
        src_points = np.float32([points[36], points[45], points[30]])
        dst_points = np.float32([[30, 30], [70, 30], [50, 70]])
        M = cv2.getAffineTransform(src_points, dst_points)
        warped = cv2.warpAffine(img, M, (100, 100))
        return warped

    def load_known_faces(self):
        """
        Loads and preprocesses known faces from the specified directory.
        """
        known_faces_dir = settings.KNOWN_FACES_DIR
        for filename in os.listdir(known_faces_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                img_path = os.path.join(known_faces_dir, filename)
                label = os.path.splitext(filename)[0]
                img = cv2.imread(img_path)
                face_features = self._preprocess_and_extract(img)
                if face_features is not None:
                    self.known_faces_features.append(face_features)
                    self.known_faces_labels.append(label)
                else:
                    print(f"Failed to extract features for known face: {label}")

    def _preprocess_and_extract(self, img):
        """
        Detects faces in an image, aligns them, and extracts features.

        Args:
            img (ndarray): The input image.

        Returns:
            ndarray: The extracted features or None if extraction fails.
        """
        faces = self._detect_faces(img)
        if faces:
            x, y, width, height = faces[0]['box']
            aligned_face = self._align_face(img, (x, y, width, height))
            face_array = self._preprocess_image(aligned_face)
            if face_array is None:
                return None
            features = self._extract_features(face_array)
            return features
        return None

    def recognize_faces(self, frame, recognition_threshold=7):
        """
        Recognizes faces in a given frame by comparing them to known faces.

        Args:
            frame (ndarray): The input frame to recognize faces in.
            recognition_threshold (float): The threshold for face recognition.

        Returns:
            list: A list of recognized faces with labels and coordinates.
        """
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        faces = self._detect_faces(gray_image_3ch)
        recognized_faces = []
        for face in faces:
            x, y, width, height = face['box']
            if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
                continue
            aligned_face = self._align_face(frame, (x, y, width, height))
            face_array = self._preprocess_image(aligned_face)
            if face_array is None:
                continue
            features = self._extract_features(face_array)
            min_distance = float('inf')
            label = 'Unknown'
            for known_features, known_label in zip(self.known_faces_features, self.known_faces_labels):
                dist = distance.euclidean(features, known_features)
                print(f"Distance to known face {known_label}: {dist}")
                if dist < min_distance:
                    min_distance = dist
                    label = known_label
            if min_distance > recognition_threshold:
                label = 'Unknown'
            face['label'] = label
            recognized_faces.append(face)

            self.save_face_image(frame[y:y + height, x:x + width], face['label'])

        return recognized_faces

    def save_face_image(self, face_img, label):
        """
        Saves the recognized face image to disk and creates a corresponding record in the database.

        Args:
            face_img (ndarray): The face image to save.
            label (str): The label of the face (e.g., name of the person).
        """
        faces_seen_dir = os.path.join(settings.MEDIA_ROOT, 'faces_seen')
        if not os.path.exists(faces_seen_dir):
            os.makedirs(faces_seen_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{label}_{timestamp}.jpg"
        filepath = os.path.join(faces_seen_dir, filename)

        cv2.imwrite(filepath, face_img)
        print(f"Face image saved: {filepath}")

        Face.objects.create(name=label, image=f"faces_seen/{filename}")
        print(f"Face record saved: {label}, {filename}")
