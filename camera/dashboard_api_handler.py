import requests
from datetime import datetime
import os
import cv2
from django.conf import settings

class DashboardAPIHandler:
    """
    A class to handle sending logs, images, and videos to a dashboard API.

    Attributes:
        api_url (str): The base URL of the dashboard API.
    """

    def __init__(self, api_url):
        """
        Initializes the DashboardAPIHandler with the provided API URL.

        Args:
            api_url (str): The base URL of the dashboard API.
        """
        self.api_url = api_url

    def send_log(self, event_type, description, extra_data=None):
        """
        Sends a log entry to the dashboard API.

        Args:
            event_type (str): The type of the event (e.g., "info", "error").
            description (str): A description of the event.
            extra_data (dict, optional): Additional data to include in the log. Defaults to None.
        """
        payload = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'description': description,
        }
        if extra_data is not None:
            payload['extra_data'] = extra_data

        try:
            response = requests.post(f"{self.api_url}/log_event/", json=payload, timeout=1)
            response.raise_for_status()
            # print("Log sent successfully")
        except requests.exceptions.RequestException as e:
            pass  # Silently fail - don't print to avoid log spam

    def send_image(self, image, description="Image uploaded"):
        """
        Sends an image to the dashboard API.

        Args:
            image (ndarray): The image to be sent, in OpenCV format.
            description (str, optional): A description of the image. Defaults to "Image uploaded".
        """
        _, img_encoded = cv2.imencode('.jpg', image)
        files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {
            'timestamp': datetime.now().isoformat(),
            'description': description
        }
        try:
            response = requests.post(f"{self.api_url}/upload_image/", files=files, data=data)
            response.raise_for_status()
            print("Image sent successfully")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send image: {e}")

    def send_video(self, video_path, description="Video snippet uploaded", thumbnail_path=None):
        """
        Sends a video file to the dashboard API, optionally with a thumbnail.

        Args:
            video_path (str): The file path to the video to be sent.
            description (str, optional): A description of the video. Defaults to "Video snippet uploaded".
            thumbnail_path (str, optional): The file path to a thumbnail image for the video. Defaults to None.
        """
        with open(video_path, 'rb') as video_file:
            files = {'video': (os.path.basename(video_path), video_file, 'video/mp4')}
            data = {
                'timestamp': datetime.now().isoformat(),
                'description': description,
                'extra_data': {}
            }
            
            if thumbnail_path:
                data['extra_data']['thumbnail'] = thumbnail_path
                print(f"Sending thumbnail: {thumbnail_path}")  # Log the thumbnail path

            try:
                response = requests.post(f"{self.api_url}/upload_video/", files=files, data=data)
                response.raise_for_status()
                print("Video sent successfully with thumbnail")
            except requests.exceptions.RequestException as e:
                print(f"Failed to send video: {e}")
