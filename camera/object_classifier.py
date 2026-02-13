"""
Object Classification module using YOLOX.

This module provides object detection and classification capabilities using YOLOX,
a high-performance anchor-free object detector. Supports both YOLOX-Small (default)
and YOLOX-Nano (for resource-constrained environments like Raspberry Pi).
"""
import os
import cv2
import numpy as np
import torch
from collections import deque
from django.conf import settings
from .device_utils import get_device

import logging

logger = logging.getLogger(__name__)

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


class ObjectClassifier:
    """
    A class used to perform object classification using YOLOX.

    Supports two model sizes:
    - 'small': YOLOX-Small (~9M params, ~40.5% mAP) - Default for GPU systems
    - 'nano': YOLOX-Nano (~0.9M params, ~25.8% mAP) - For edge devices like RPi

    Attributes:
        device (torch.device): The compute device (CUDA or CPU).
        model: The YOLOX model instance.
        model_size (str): The model size being used ('small' or 'nano').
        confidence_threshold (float): Minimum confidence for detections.
        nms_threshold (float): Non-maximum suppression threshold.
        prediction_buffer (deque): Buffer for smoothing predictions.
        class_names (list): List of COCO class names.
    """

    def __init__(self, model_size='small', buffer_size=15, confidence_threshold=0.5,
                 nms_threshold=0.45, device=None):
        """
        Initializes the ObjectClassifier with YOLOX model.

        Args:
            model_size (str): Model size - 'small' (default) or 'nano' (for RPi).
            buffer_size (int): Size of prediction smoothing buffer.
            confidence_threshold (float): Minimum confidence for predictions.
            nms_threshold (float): NMS threshold for filtering overlapping boxes.
            device: Optional torch.device. If None, auto-detects CUDA/CPU.
        """
        self.device = device if device is not None else get_device()
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.class_names = COCO_CLASSES
        
        # Prediction smoothing buffer
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

        # Input size for YOLOX
        self.input_size = (640, 640) if model_size == 'small' else (416, 416)

        # Load the model
        self.model = self._load_model()
        
        logger.info(f"ObjectClassifier initialized with YOLOX-{model_size} on {self.device}")

    def _load_model(self):
        """
        Loads the YOLOX model from torch hub or local weights.

        Returns:
            The loaded YOLOX model in evaluation mode.
        """
        try:
            # Try loading from torch hub first
            model_name = f'yolox_{self.model_size}'
            
            # Check for local ONNX models first (from existing yolo folder)
            local_model_dir = os.path.join(settings.MODEL_DIR, 'yolo')
            
            # Try to use torch hub YOLOX
            try:
                model = torch.hub.load('Megvii-BaseDetection/YOLOX', model_name, pretrained=True)
                model = model.to(self.device)
                model.eval()
                logger.info(f"Loaded YOLOX-{self.model_size} from torch hub")
                return model
            except Exception as hub_error:
                logger.warning(f"Failed to load from torch hub: {hub_error}")
                
                # Fallback: Use a simpler YOLOv5 from ultralytics (more reliable hub)
                logger.info("Falling back to YOLOv5 from ultralytics")
                try:
                    # Force CPU to avoid CUDA incompatibility issues
                    model = torch.hub.load('ultralytics/yolov5', 
                                          'yolov5s' if self.model_size == 'small' else 'yolov5n',
                                          pretrained=True,
                                          device='cpu')  # Force CPU loading
                    # Only move to device if it's CPU or if CUDA actually works
                    if self.device.type == 'cpu':
                        model = model.to('cpu')
                    else:
                        try:
                            model = model.to(self.device)
                        except RuntimeError:
                            logger.warning("CUDA incompatible, keeping model on CPU")
                            self.device = torch.device('cpu')
                            model = model.to('cpu')
                    model.eval()
                    self._using_yolov5 = True
                    logger.info(f"Loaded YOLOv5-{'s' if self.model_size == 'small' else 'n'} as fallback on {self.device}")
                    return model
                except Exception as yolov5_error:
                    logger.error(f"YOLOv5 fallback also failed: {yolov5_error}")
                    raise

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Return a dummy model that won't crash
            self._using_fallback = True
            return None

    def _preprocess(self, image):
        """
        Preprocesses an image for YOLOX inference.

        Args:
            image (ndarray): Input BGR image.

        Returns:
            tuple: (preprocessed_tensor, scale_ratio) for inference.
        """
        if image is None or image.size == 0:
            return None, 1.0

        # Get original dimensions
        h, w = image.shape[:2]
        
        # Calculate scale to fit input size while maintaining aspect ratio
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        padded[:new_h, :new_w] = resized
        
        # Convert to tensor
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor, scale

    def _postprocess(self, outputs, scale, original_shape):
        """
        Post-processes YOLOX outputs to get detections.

        Args:
            outputs: Raw model outputs.
            scale: Scale ratio used during preprocessing.
            original_shape: Original image shape (h, w).

        Returns:
            list: List of detections, each with class_id, confidence, and bbox.
        """
        detections = []
        
        if outputs is None:
            return detections

        # Handle YOLOv5 output format (from fallback)
        if hasattr(self, '_using_yolov5') and self._using_yolov5:
            # YOLOv5 returns a Results object
            if hasattr(outputs, 'xyxy'):
                for det in outputs.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2, conf, cls_id = det
                    if conf >= self.confidence_threshold:
                        detections.append({
                            'class_id': int(cls_id),
                            'class_name': self.class_names[int(cls_id)] if int(cls_id) < len(self.class_names) else 'unknown',
                            'confidence': float(conf),
                            'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        })
            return detections

        # Handle standard YOLOX output
        try:
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu().numpy()
            
            for output in outputs:
                if output is None:
                    continue
                    
                for det in output:
                    confidence = det[4]
                    if confidence < self.confidence_threshold:
                        continue
                    
                    class_scores = det[5:]
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    
                    if class_conf * confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = det[:4] / scale
                        detections.append({
                            'class_id': int(class_id),
                            'class_name': self.class_names[int(class_id)] if int(class_id) < len(self.class_names) else 'unknown',
                            'confidence': float(confidence * class_conf),
                            'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        })
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")

        return detections

    def detect_objects(self, image):
        """
        Detects objects in an image and returns all detections.

        Args:
            image (ndarray): Input BGR image.

        Returns:
            list: List of detection dicts with class_name, confidence, bbox.
        """
        if self.model is None:
            return []

        try:
            # Handle YOLOv5 fallback (simpler API)
            if hasattr(self, '_using_yolov5') and self._using_yolov5:
                # YOLOv5 handles preprocessing internally
                results = self.model(image)
                return self._postprocess(results, 1.0, image.shape[:2])

            # Standard YOLOX inference
            img_tensor, scale = self._preprocess(image)
            if img_tensor is None:
                return []

            with torch.no_grad():
                outputs = self.model(img_tensor)

            detections = self._postprocess(outputs, scale, image.shape[:2])
            return detections

        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []

    def classify_object(self, image):
        """
        Classifies objects in the provided image using YOLOX.
        Returns the most confident detection, smoothed over the prediction buffer.

        This method maintains backward compatibility with the original interface.

        Args:
            image (ndarray): The image in which objects need to be classified.

        Returns:
            str: The label of the detected object with the highest average confidence.
                 Returns 'unknown' if no confident prediction is made.
        """
        detections = self.detect_objects(image)
        
        # Build predictions dict for this frame
        predictions = {}
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            if class_name not in predictions or confidence > predictions[class_name]:
                predictions[class_name] = confidence

        # Add to buffer
        self.prediction_buffer.append(predictions)

        # Smooth predictions over buffer
        averaged_predictions = {class_name: 0.0 for class_name in self.class_names}
        for preds in self.prediction_buffer:
            for class_name, conf in preds.items():
                if class_name in averaged_predictions:
                    averaged_predictions[class_name] += conf

        # Find the class with highest averaged confidence
        if averaged_predictions:
            final_label = max(averaged_predictions, key=averaged_predictions.get)
            if averaged_predictions[final_label] > 0:
                return final_label

        return 'unknown'

    def annotate_image(self, image, text, position=(10, 50), font=cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale=1, color=(255, 255, 255), thickness=2):
        """
        Annotates the image with the specified text.

        Args:
            image (ndarray): The image frame to annotate.
            text (str): The text to annotate on the image.
            position (tuple): Position (x, y) for the text.
            font (int): Font type for the text.
            font_scale (float): Scale of the text font.
            color (tuple): Color of the text in (B, G, R) format.
            thickness (int): Thickness of the text.
        """
        cv2.putText(image, text, position, font, font_scale, color, thickness)

    def draw_detections(self, image, detections=None):
        """
        Draws detection boxes and labels on the image.

        Args:
            image (ndarray): The image to annotate.
            detections (list): Optional list of detections. If None, runs detection.

        Returns:
            ndarray: Annotated image.
        """
        if detections is None:
            detections = self.detect_objects(image)

        for det in detections:
            x, y, w, h = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x, y - text_h - 10), (x + text_w, y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    def set_model_size(self, model_size):
        """
        Changes the model size and reloads the model.
        Useful for switching between 'small' and 'nano' at runtime.

        Args:
            model_size (str): New model size ('small' or 'nano').
        """
        if model_size not in ('small', 'nano'):
            logger.warning(f"Invalid model size: {model_size}. Using 'small'.")
            model_size = 'small'

        if model_size != self.model_size:
            self.model_size = model_size
            self.input_size = (640, 640) if model_size == 'small' else (416, 416)
            self.model = self._load_model()
            logger.info(f"Switched to YOLOX-{model_size}")
