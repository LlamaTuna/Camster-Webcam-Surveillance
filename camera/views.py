import threading
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.core.paginator import Paginator
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import login
from .models import Face, Event
from .forms import TagFaceForm, CustomUserCreationForm, UploadFaceForm
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from scipy.spatial.distance import euclidean
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
from datetime import datetime, timezone, timedelta
from .utils import reconcile_faces
import pytz
import logging
from .video_camera import VideoCamera
from .forms import EmailSettingsForm, UserSettingsForm, RecognitionSettingsForm
from .models import EmailSettings, RecognitionSettings
from urllib.parse import unquote
from django.core.cache import cache
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import LogSerializer
from .forms import AudioDeviceSettingForm
from .models import AudioDeviceSetting

import sys
import torch
from .device_utils import get_device

camera_instances = []

log_lock = threading.Lock()
logs = []

# Global variable to hold the camera instance
camera_instance = None

def list_cameras(max_cameras=4):
    """
    Lists available camera devices, prioritizing /dev/video0, and limits the list to a maximum of 4 devices.
    
    Args:
        max_cameras (int): The maximum number of camera devices to return.
    
    Returns:
        list: A list of paths to the available camera devices, limited to max_cameras.
    """
    camera_devices = []
    
    try:
        # Check if /dev/video0 exists and add it first
        if os.path.exists('/dev/video0'):
            camera_devices.append('/dev/video0')
        
        # List other video devices, excluding /dev/video0
        for filename in os.listdir('/dev'):
            if filename.startswith('video') and f'/dev/{filename}' != '/dev/video0':
                device_path = os.path.join('/dev', filename)
                camera_devices.append(device_path)
        
        # Limit the number of cameras to max_cameras
        camera_devices = camera_devices[:max_cameras]
        
    except Exception as e:
        print(f"Error listing cameras: {e}")
    
    print("Camera devices:", camera_devices)
    return camera_devices

list_cameras()

def reload_all_known_faces():
    """
    Reloads known faces in all active camera instances.
    Called after tagging a new face to update recognition immediately.
    """
    global camera_instances
    for camera in camera_instances:
        try:
            if hasattr(camera, 'facial_recognition'):
                camera.facial_recognition.reload_known_faces()
                print(f"Reloaded known faces for camera {camera.camera_index}")
        except Exception as e:
            print(f"Error reloading known faces: {e}")

def log_event(event):
    """
    Logs an event with a timestamp.

    Args:
        event (str): Description of the event.
    """
    global logs
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {event}"
    with log_lock:
        logs.append(log_entry)
    print("log event call", log_entry)  # Debug statement

def get_logs(request):
    """
    Returns the last 100 log entries as a JSON response.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: A JSON response containing the last 100 log entries.
    """
    global logs
    with log_lock:
        log_data = logs[-100:]  # Get the last 100 log entries
    print("Fetching logs:", log_data)  # Debug statement
    return JsonResponse({'logs': log_data})

def initialize_camera(request, device_path):
    """
    Initializes and returns a VideoCamera instance for the specified device path.

    Args:
        request (HttpRequest): The HTTP request object.
        device_path (str): The path to the camera device.

    Returns:
        VideoCamera: The initialized VideoCamera instance, or None if initialization fails.
    """
    global camera_instances
    normalized_device_path = f"/dev/{device_path.split('/')[-1]}"

    # Check if the camera is already initialized
    for camera in camera_instances:
        if camera.camera_index == normalized_device_path:
            print(f"Camera at {normalized_device_path} is already initialized.")
            return camera

    # Create new camera instance if not found in initialized list
    camera = VideoCamera(camera_index=normalized_device_path, request=request)
    if camera.video is None or not camera.video.isOpened():
        print(f"Failed to open camera at {normalized_device_path}.")
        return None

    camera_instances.append(camera)
    print(f"Camera at {normalized_device_path} initialized successfully.")
    return camera

def initialize_all_cameras(request):
    """
    Initializes VideoCamera instances for all available cameras.

    Args:
        request (HttpRequest): The HTTP request object.
    """
    global camera_instances
    camera_devices = list_cameras()
    for device_path in camera_devices:
        initialize_camera(request, device_path)

def gen(camera):
    """
    Generator function to yield frames from the camera.

    Args:
        camera (VideoCamera): The camera instance.

    Yields:
        bytes: JPEG-encoded frame.
    """
    import time
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            # Small sleep to prevent tight loop when frame skipping
            time.sleep(0.01)

def video_feed(request, device_path):
    """
    Streams the video feed from the specified camera device.

    Args:
        request (HttpRequest): The HTTP request object.
        device_path (str): The path to the camera device.

    Returns:
        StreamingHttpResponse: The video stream from the camera.
    """
    global camera_instances

    # Normalize device path (remove leading/trailing slashes)
    normalized_device_path = f"/dev/{device_path.split('/')[-1]}"

    # Check if camera instance for this device path already exists
    for i, camera in enumerate(camera_instances):
        if camera.camera_index == normalized_device_path:
            # Check if the camera instance is still valid (executor not shut down)
            try:
                # Test if we can read a frame - if video is None or executor shut down, recreate
                if camera.video is None or not camera.video.isOpened():
                    raise RuntimeError("Camera video capture is invalid")
                print(f"Reusing existing camera instance for {normalized_device_path}")
                return StreamingHttpResponse(gen(camera),
                                             content_type='multipart/x-mixed-replace; boundary=frame')
            except (RuntimeError, Exception) as e:
                print(f"Camera instance for {normalized_device_path} is stale ({e}), recreating...")
                camera_instances.pop(i)
                break

    # If not found or stale, create and cache a new camera instance
    print(f"Creating new camera instance for {normalized_device_path}")
    camera = VideoCamera(camera_index=normalized_device_path, request=request)
    if camera.video is None or not camera.video.isOpened():
        return HttpResponse("Camera not found", status=404)

    camera_instances.append(camera)

    return StreamingHttpResponse(gen(camera),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    """
    Renders the index page with available camera device paths.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered index page with available camera device paths.
    """
    camera_devices = list_cameras()
    initialized_cameras = []

    for device_path in camera_devices:
        camera = initialize_camera(request, device_path)
        if camera:  # Only add if initialization was successful
            initialized_cameras.append(device_path)

    context = {'camera_devices': initialized_cameras}

    # Show welcome message after registration
    welcome_user = request.GET.get('welcome')
    if welcome_user:
        context['welcome_message'] = f'Welcome, {welcome_user}! Your account has been created and you are now logged in.'

    return render(request, 'camera/index.html', context)

def camera_view(request, device_path):
    """
    Handles the camera view for the specified device path.

    Args:
        request (HttpRequest): The HTTP request object.
        device_path (str): The path to the camera device.

    Returns:
        HttpResponse: The rendered camera view page.
    """
    # Initialize all cameras if not already done (you might want to optimize this)
    if not camera_instances:
        initialize_all_cameras(request)

    # Render the camera view template
    return render(request, 'camera_view.html', {'device_path': device_path})

@login_required
def list_faces(request):
    """
    Lists faces with pagination and filtering.

    Query params:
        filter: 'all', 'tagged', 'untagged' (default: 'all')
        page: page number (default: 1)

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered list_faces page with paginated faces.
    """
    reconcile_faces()  # Reconcile the database with the actual images

    # Filter logic
    filter_param = request.GET.get('filter', 'all')
    if filter_param == 'tagged':
        faces_qs = Face.objects.filter(tagged=True).order_by('-timestamp')
    elif filter_param == 'untagged':
        faces_qs = Face.objects.filter(tagged=False).order_by('-timestamp')
    else:
        faces_qs = Face.objects.all().order_by('-timestamp')

    # Counts for filter badges
    total_count = Face.objects.count()
    tagged_count = Face.objects.filter(tagged=True).count()
    untagged_count = Face.objects.filter(tagged=False).count()

    # Pagination (20 per page)
    paginator = Paginator(faces_qs, 20)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    return render(request, 'camera/list_faces.html', {
        'faces': page_obj,
        'page_obj': page_obj,
        'current_filter': filter_param,
        'total_count': total_count,
        'tagged_count': tagged_count,
        'untagged_count': untagged_count,
    })

@login_required
def tag_face(request, face_id):
    """
    Tags a face with the provided details.

    Args:
        request (HttpRequest): The HTTP request object.
        face_id (int): The ID of the face to be tagged.

    Returns:
        HttpResponse: The rendered tag_face page with the form or a redirect to list_faces.
    """
    face = Face.objects.get(id=face_id)
    if request.method == 'POST':
        form = TagFaceForm(request.POST, request.FILES, instance=face)
        if form.is_valid():
            form.save()
            known_faces_path = settings.KNOWN_FACES_DIR
            print("Known Faces Path:", known_faces_path)  # Debug statement
            if not os.path.exists(known_faces_path):
                os.makedirs(known_faces_path)
            new_path = os.path.join(known_faces_path, os.path.basename(face.image.path))
            print("New Path:", new_path)  # Debug statement
            os.rename(face.image.path, new_path)
            face.image.name = os.path.join('known_faces', os.path.basename(new_path))
            face.tagged = True
            
            # Compute and store embedding for the tagged face
            try:
                from .facial_recognition import FacialRecognition
                fr = FacialRecognition()
                img = cv2.imread(new_path)
                if img is not None:
                    boxes, probs, face_tensors = fr._detect_faces(img)
                    if len(face_tensors) > 0:
                        features = fr._extract_features(face_tensors[0])
                        if features is not None:
                            import numpy as np
                            face.embedding = features.astype(np.float32).tobytes()
                            print(f"Computed and stored embedding for {face.name}")
            except Exception as e:
                print(f"Error computing embedding: {e}")
            
            face.save()
            
            # Reload known faces in all active camera instances
            reload_all_known_faces()
            
            return redirect('list_faces')
    else:
        form = TagFaceForm(instance=face)
    return render(request, 'camera/tag_face.html', {'form': form})

@staff_member_required
def admin_view(request):
    """
    Renders the admin view page.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered admin view page.
    """
    return render(request, 'camera/admin.html')

def register(request):
    """
    Handles user registration.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered register page with the form or a redirect to index.
    """
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect(f'/?welcome={user.username}')
    else:
        form = CustomUserCreationForm()
    return render(request, 'camera/register.html', {'form': form})

@login_required
def upload_face(request):
    """
    Handles the upload of a new face image.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered upload_face page with the form or a redirect to list_faces.
    """
    if request.method == 'POST':
        form = UploadFaceForm(request.POST, request.FILES)
        if form.is_valid():
            face = form.save(commit=False)
            # Process the uploaded image
            image_file = request.FILES['image']
            image_array = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                form.add_error('image', 'Image not valid. Please upload a valid image file.')
            else:
                # Detect and crop the face using PyTorch MTCNN
                device = get_device()
                detector = MTCNN(keep_all=False, device=device)
                
                # Convert BGR to RGB for MTCNN
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes, probs = detector.detect(rgb_image)
                
                if boxes is not None and len(boxes) > 0:
                    # Get the first face bounding box [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [int(coord) for coord in boxes[0]]
                    width, height = x2 - x1, y2 - y1
                    x, y = x1, y1
                    cropped_face = image[y:y + height, x:x + width]

                    # Ensure the known_faces directory exists
                    known_faces_dir = settings.KNOWN_FACES_DIR
                    if not os.path.exists(known_faces_dir):
                        os.makedirs(known_faces_dir)

                    # Create a unique filename for the cropped face
                    filename = f"{face.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cropped_path = os.path.join(known_faces_dir, filename)
                    
                    # Debugging statements
                    print(f"Known Faces Directory: {known_faces_dir}")
                    print(f"Filename: {filename}")
                    print(f"Cropped Path: {cropped_path}")

                    # Save cropped face directly to known_faces directory
                    cv2.imwrite(cropped_path, cropped_face)

                    # Update face image path and save the record
                    face.image.name = os.path.join('known_faces', filename)
                    face.tagged = True
                    
                    # Compute and store embedding
                    try:
                        from .facial_recognition import FacialRecognition
                        from facenet_pytorch import InceptionResnetV1
                        
                        fr = FacialRecognition()
                        img = cv2.imread(cropped_path)
                        if img is not None:
                            boxes_fr, probs_fr, face_tensors = fr._detect_faces(img)
                            if len(face_tensors) > 0:
                                features = fr._extract_features(face_tensors[0])
                                if features is not None:
                                    face.embedding = features.astype(np.float32).tobytes()
                                    print(f"Computed and stored embedding for {face.name}")
                    except Exception as e:
                        print(f"Error computing embedding: {e}")
                    
                    face.save()
                    
                    # Reload known faces in all active camera instances
                    reload_all_known_faces()

                    return redirect('list_faces')
                else:
                    form.add_error('image', 'No face detected in the image. Please upload a different image.')
    else:
        form = UploadFaceForm()
    return render(request, 'camera/upload_face.html', {'form': form})

@login_required
def email_settings(request):
    """
    Handles the update of email settings for the logged-in user.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered email_settings page with the form or a redirect to email_settings.
    """
    try:
        email_settings = EmailSettings.objects.get(user=request.user)
    except EmailSettings.DoesNotExist:
        email_settings = None

    if request.method == 'POST':
        form = EmailSettingsForm(request.POST, instance=email_settings)
        if form.is_valid():
            email_settings = form.save(commit=False)
            email_settings.user = request.user
            email_settings.save()
            print("Email settings saved:", email_settings.__dict__)  # Debug statement
            return redirect('email_settings')
    else:
        form = EmailSettingsForm(instance=email_settings)
    
    return render(request, 'camera/email_settings.html', {'form': form})

@login_required
def user_settings(request):
    """
    Handles the update of user settings for the logged-in user.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered user_settings page with the form or a redirect to user_settings.
    """
    if request.method == 'POST':
        form = UserSettingsForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('user_settings')
    else:
        form = UserSettingsForm(instance=request.user)
    return render(request, 'camera/user_settings.html', {'form': form})

@login_required
def recognition_settings(request):
    """
    Handles the update of face recognition settings (similarity threshold).

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered recognition_settings page with the form.
    """
    settings_obj = RecognitionSettings.get_settings()
    
    if request.method == 'POST':
        form = RecognitionSettingsForm(request.POST, instance=settings_obj)
        if form.is_valid():
            form.save()
            # Reload known faces to apply new threshold
            reload_all_known_faces()
            return redirect('recognition_settings')
    else:
        form = RecognitionSettingsForm(instance=settings_obj)
    
    # Get current known faces count
    known_faces_count = Face.objects.filter(tagged=True).count()
    
    return render(request, 'camera/recognition_settings.html', {
        'form': form,
        'known_faces_count': known_faces_count
    })

@login_required
def delete_all_faces(request):
    """
    Deletes all face records and their associated image files.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: A redirect to the list_faces page after deletion.
    """
    faces = Face.objects.all()
    for face in faces:
        if face.image:
            if os.path.isfile(face.image.path):
                os.remove(face.image.path)  # Delete the image file from the filesystem
        face.delete()  # Delete the database record
    
    return redirect('list_faces')

@api_view(['POST'])
def log_event(request):
    """
    API endpoint to log an event.

    Args:
        request (HttpRequest): The HTTP request object containing event data.

    Returns:
        Response: A response indicating whether the log was received or an error occurred.
    """
    serializer = LogSerializer(data=request.data)
    if serializer.is_valid():
        return Response({"message": "Log received"}, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@login_required
def device_settings(request):
    """
    Handles the selection of audio devices for camera devices.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered device_settings page with the form or a redirect to device_settings.
    """
    available_devices = list_cameras()  # This gets the available camera devices

    # If no camera devices are found, you might want to handle this case
    if not available_devices:
        return render(request, 'camera/device_settings.html', {'error': 'No camera devices found.'})

    # Automatically select the first available camera device
    selected_device = available_devices[0]  # Select the first camera device as a default

    if request.method == 'POST':
        form = AudioDeviceSettingForm(request.POST)
        if form.is_valid():
            setting = form.save(commit=False)
            setting.user = request.user
            setting.camera_index = selected_device  # Automatically set the camera_index based on available devices
            setting.save()
            print("ATTN! Audio device setting saved:", selected_device + " - " + setting.audio_device)  # Debug statement
            return redirect('device_settings')
    else:
        form = AudioDeviceSettingForm(initial={'camera_index': selected_device})

    return render(request, 'camera/device_settings.html', {'form': form})
