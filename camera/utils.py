# utils.py
import os
import logging
from django.conf import settings
from .models import Face
import threading
from datetime import datetime
from django.http import JsonResponse

logger = logging.getLogger(__name__)

log_lock = threading.Lock()
logs = []

def reconcile_faces():
    """
    Reconciles the face records in the database with the actual images in the file system.

    Checks each face record in the database to see if the corresponding image file exists.
    If the image file does not exist, the face record is deleted from the database.
    """
    faces = Face.objects.all()
    faces_seen_dir = os.path.join(settings.MEDIA_ROOT, 'faces_seen')

    for face in faces:
        image_path = os.path.join(settings.MEDIA_ROOT, face.image.name)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}. Deleting record from database.")
            face.delete()

def log_event(event):
    """
    Logs an event with a timestamp.

    The event is stored in a global logs list, with each entry containing the event description and a timestamp.

    Args:
        event (str): The event description to be logged.
    """
    global logs
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {event}"
    with log_lock:
        logs.append(log_entry)
    print("log event call", log_entry)  # Debug statement

def get_logs(request):
    """
    Retrieves the last 100 log entries.

    This function returns a JSON response containing the most recent 100 log entries.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: A JSON response containing the log entries.
    """
    global logs
    with log_lock:
        log_data = logs[-100:]  # Get the last 100 log entries
    print("Fetching logs:", log_data)  # Debug statement
    return JsonResponse({'logs': log_data})