import threading
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from .movement_detection import MovementDetection
from .facial_recognition import FacialRecognition
from .send_email import SendEmail
import pytz
import os
from django.conf import settings
from .models import Event, AudioDeviceSetting
import subprocess
from .object_classifier import ObjectClassifier
from .dashboard_api_handler import DashboardAPIHandler
from .audio_source import AudioSource


class VideoCamera:
    """
    A class to manage video streaming, frame processing, and event handling from a camera device, including audio capture using ALSA.

    Attributes:
        camera_index (int): The index of the camera device to use.
        resolution (tuple): The resolution of the video feed.
        request (HttpRequest): The request object, used to access user-specific settings.
        video (cv2.VideoCapture): The OpenCV video capture object.
        initialized (bool): A flag indicating whether the camera was successfully initialized.
        audio_source (AudioSource): Manages ALSA audio source selection and capture.
        audio_device (str): The selected ALSA audio device for recording.
        movement_detection (MovementDetection): Handles movement detection in frames.
        facial_recognition (FacialRecognition): Handles facial recognition.
        send_email (SendEmail): Handles sending alert emails.
        dashboard_api (DashboardAPIHandler): Handles sending logs and video clips to a dashboard API.
        object_classifier (ObjectClassifier): Handles object classification in frames.
        classification_interval (int): The number of frames between object classifications.
        classification_counter (int): A counter to track frames for classification.
        frame_skip_interval (int): The number of frames to skip between processing.
        frame_count (int): A counter to track frames processed.
        face_recognition_interval (int): The number of frames between face recognition.
        face_recognition_counter (int): A counter to track frames for face recognition.
        lock (threading.Lock): A lock to synchronize access to shared resources.
        frames (list): A list to store frames for processing.
        detected_faces (list): A list to store detected faces in frames.
        executor (ThreadPoolExecutor): An executor to manage background tasks for frame processing.
        email_executor (ThreadPoolExecutor): An executor to manage background tasks for email sending.
        save_timer (threading.Timer): A timer to save running buffer clips periodically.
        frame_buffer (list): A buffer to store frames for email snapshots.
        running_buffer (list): A buffer to store frames for creating video clips.
        last_alert_time (float): The timestamp of the last alert sent.
        alert_interval (int): The minimum time interval between alerts.
    """

    def __init__(self, camera_index=0, resolution=(320, 240), request=None):
        """
        Initializes the VideoCamera object, setting up video capture, audio management,
        movement detection, facial recognition, and object classification.

        Args:
            camera_index (int): The index of the camera device to use.
            resolution (tuple): The resolution of the video feed.
            request (HttpRequest): The request object, used to access user-specific settings.
        """

            
        # Initialize the audio source, will fall back if no usable audio device
       # Initialize audio source and add event listener
        self.resolution = resolution  
        self.audio_source = AudioSource()
        self.audio_source.add_listener(self.on_audio_event)  # Capture audio events
        self.audio_source.start()


        self.camera_index = camera_index
        self.video = cv2.VideoCapture(camera_index)
        if not self.video.isOpened():
            print(f"Error: Could not open video device at {camera_index}.")
            self.video = None
            self.initialized = False  # Camera failed to open
            return

        self.initialized = True  # Camera successfully opened
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.movement_detection = MovementDetection()
        self.facial_recognition = FacialRecognition()
        self.send_email = SendEmail(request)

        self.dashboard_api = DashboardAPIHandler(settings.DASHBOARD_API_URL)

        self.object_classifier = ObjectClassifier()  # Instantiate ObjectClassifier
        self.classification_interval = 5  # Classify every 5 frames
        self.classification_counter = 0

        self.frame_skip_interval = 2
        self.frame_count = 0
        self.face_recognition_interval = 10
        self.face_recognition_counter = 0

        self.lock = threading.Lock()
        self.frames = []
        self.detected_faces = []

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(self._process_frames)

        self.async_executor = ThreadPoolExecutor(max_workers=2)  # For classification and API calls
        self.email_executor = ThreadPoolExecutor(max_workers=1)  # Executor for email sending

        self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
        self.save_timer.start()

        self.frame_buffer = []
        self.running_buffer = []
        self.last_alert_time = time.time()
        self.alert_interval = 30  # 30 seconds

    def on_audio_event(self, volume):
        """Triggered when audio event occurs. Capture frame and store it."""
        print(f"Audio event detected with volume: {volume}. Capturing frame...")
        success, frame = self.video.read()
        timestamp = time.time()  # Capture the exact time when the frame is captured
        if success:
            self.frames.append((frame, timestamp))
            self.running_buffer.append((frame, timestamp))  # Store frame and timestamp

    
    def __del__(self):
        """Handles cleanup by releasing resources when the object is destroyed."""
        if self.video:
            self.video.release()
        if hasattr(self, 'save_timer'):
            self.save_timer.cancel()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        if hasattr(self, 'async_executor'):
            self.async_executor.shutdown(wait=False)
        if hasattr(self, 'email_executor'):
            self.email_executor.shutdown(wait=False)
        if hasattr(self, 'pulse_manager') and self.pulse_manager:
            self.pulse_manager.close()
            
    def get_frame(self):
        """
        Captures a frame from the video feed, processes it for movement detection,
        face recognition, and object classification, and returns the processed frame.

        Returns:
            bytes: The processed frame as a JPEG-encoded image, or None if capturing failed.
        """
        if not self.video:
            return None
        success, image = self.video.read()
        if not success:
            return None

        self.frame_count += 1
        if self.frame_count % self.frame_skip_interval != 0:
            return None

        image = cv2.resize(image, (320, 240))

        movement_detected, movement_box = self.movement_detection.detect_movement(image)
        if movement_detected:
            with self.lock:
                self.frames.append(image.copy())
            x, y, width, height = movement_box
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 1)
            cv2.putText(image, "Movement Detected", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
            self.frame_buffer.append(image.copy())  # Ensure frame is added here
            self.running_buffer.append(image.copy())

            # Only classify objects if movement is detected
            try:
                self.async_executor.submit(self.dashboard_api.send_log, "movement", "Movement detected", {"movement_box": movement_box})
            except RuntimeError:
                pass  # Executor shut down
            self.classification_counter += 1
            if self.classification_counter >= self.classification_interval:
                self.classification_counter = 0
                # Run classification asynchronously to not block the stream
                try:
                    if not hasattr(self, '_classification_running'):
                        self._classification_running = False
                        self._last_classification = None
                    
                    if not self._classification_running:
                        self._classification_running = True
                        def run_classification(img):
                            try:
                                result = self.object_classifier.classify_object(img)
                                self._last_classification = result
                                print(f"{result} seen in the frame")
                                self.dashboard_api.send_log("classification", f"{result} seen in the frame")
                                self.send_email.log_event(f"{result} seen in the frame")
                            except Exception as e:
                                print(f"Classification error: {e}")
                            finally:
                                self._classification_running = False
                        
                        self.async_executor.submit(run_classification, image.copy())
                except RuntimeError:
                    pass  # Executor shut down
            
            # Display last classification result if available
            if hasattr(self, '_last_classification') and self._last_classification:
                cv2.putText(image, self._last_classification, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Attempt to send email snapshot
            if time.time() - self.last_alert_time >= self.alert_interval:
                self.send_email.log_event("Movement detected")
                self.send_email.frame_buffer = self.frame_buffer.copy()
                try:
                    self.email_executor.submit(self.send_email.send_email_snapshot)  # Send email asynchronously
                except RuntimeError:
                    # Executor was shut down (e.g., during server reload), skip email
                    pass
                print("Email sent from VC class")
                self.last_alert_time = time.time()

        detected_faces = []
        with self.lock:
            for face in self.detected_faces:
                detected_faces.append(face)

        for face in detected_faces:
            x, y, width, height = face['box']
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 1)
            label = face.get('label', 'Unknown')
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)

        try:
            local_time = datetime.now(pytz.timezone('US/Pacific'))
            timestamp_text = local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            cv2.putText(image, timestamp_text, (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error adding timestamp: {e}")

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def _process_frames(self):
        """
        Background task that processes frames for face recognition and updates
        the list of detected faces.
        """
        while True:
            if not self.frames:
                time.sleep(0.01)
                continue

            with self.lock:
                frame = self.frames.pop(0)

            self.face_recognition_counter += 1
            if self.face_recognition_counter >= self.face_recognition_interval:
                recognized_faces = self.facial_recognition.recognize_faces(frame)
                with self.lock:
                    self.detected_faces = recognized_faces
                self.send_email.set_detected_faces(recognized_faces)  # Pass detected faces to SendEmail
                self.face_recognition_counter = 0

                # Send face recognition log with face names
                for face in recognized_faces:
                    face_name = face.get('label', 'Unknown')
                    self.dashboard_api.send_log("face_recognition", f"Detected face: {face_name}", extra_data={"face_name": face_name})

    def save_running_buffer_clip(self):
        """
        Saves the frames in the running buffer as a video clip, captures audio, generates a thumbnail,
        and sends the clip and thumbnail to the dashboard API and via email.
        """
        # Skip save if buffer has too few frames (prevents empty/corrupt clips)
        if len(self.running_buffer) < 5:
            print(f"Skipping clip save: buffer has only {len(self.running_buffer)} frames")
            self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
            self.save_timer.start()
            return

        # Directories for saving clips and thumbnails
        event_clips_dir = os.path.join(settings.MEDIA_ROOT, 'event_clips')
        thumbnails_dir = os.path.join(settings.MEDIA_ROOT, 'thumbnails')

        if not os.path.exists(event_clips_dir):
            os.makedirs(event_clips_dir)
        if not os.path.exists(thumbnails_dir):
            os.makedirs(thumbnails_dir)

        # Timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f"event_{timestamp}.mp4"
        video_file_path = os.path.join(event_clips_dir, video_filename)

        # Frame rate and duration
        fps = 15  # Frames per second
        duration_seconds = len(self.running_buffer) / fps  # Calculate duration from the number of frames in the buffer

        # FFmpeg command to handle both video and audio creation with sync options
        command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.resolution[0]}x{self.resolution[1]}',  # Video resolution
            '-r', str(fps),  # Frame rate
            '-i', '-',  # Input comes from a pipe (for video frames)
        ]

        # Capture audio from the AudioSource
        audio_device = self.audio_source.get_device_name()
        if audio_device and audio_device != 'default':
            command.extend([
                '-itsoffset', '10',  # adjust the offset here if needed for synchronization
                '-f', 'alsa',  # Use ALSA for audio input
                '-i', f'{audio_device}',  # ALSA device for audio input
                '-c:v', 'libx264',  # Video codec
                '-preset', 'fast',  # Video encoding speed
                '-c:a', 'aac',  # Audio codec
                '-ar', '48000',  # Audio sample rate
                '-b:a', '128k',  # Audio bitrate
                '-pix_fmt', 'yuv420p',  # Pixel format for video
                '-async', '1',  # Sync the audio stream with the video
                '-vsync', '1',  # Keep video in sync with the audio
            ])
        else:
            # If no audio is available, proceed without adding audio settings
            command.extend([
                '-c:v', 'libx264',  # Video codec
                '-preset', 'fast',  # Video encoding speed
                '-pix_fmt', 'yuv420p',  # Pixel format for video
                '-vsync', '1',  # Keep video in sync
            ])
        command.extend([
            '-vf', 'setpts=PTS*1.5',  # Slow down the playback by 2x
        ])


        command.extend([
            '-t', str(duration_seconds),  # Specify the duration of the video
            '-movflags', '+faststart',  # Move moov atom to front for native player compatibility
            '-f', 'mp4',  # Specify MP4 as the output format
            video_file_path  # Output video file path
        ])


        # Start FFmpeg process
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        try:
            # Write all frames from the running buffer to FFmpeg
            for frame in self.running_buffer:
                process.stdin.write(frame.tobytes())

        except Exception as e:
            print(f"Error writing frame to FFmpeg process: {e}")

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            error_output = process.stderr.read().decode()
            print(f"FFmpeg error: {error_output}")
        else:
            # Ensure the file is fully written and closed before sending
            print(f"Video file {video_file_path} written successfully")

            # Change permissions and/or ownership after the file is created
            try:
                # Change file permissions to allow read/write access
                os.chmod(video_file_path, 0o666)  # rw-rw-rw-
                # Optionally, change file ownership (replace 'your-username' with the actual user)
                # os.chown(video_file_path, uid, gid)
                print(f"Permissions changed for {video_file_path}")
            except Exception as e:
                print(f"Error changing permissions for {video_file_path}: {e}")

            # Check for file size stabilization
            wait_for_file_stabilization(video_file_path)
            # Attempt to generate a thumbnail
            try:
                thumbnail_filename = f"thumb_{timestamp}.jpg"
                thumbnail_path = os.path.join(thumbnails_dir, thumbnail_filename)
                self.generate_thumbnail(video_file_path, thumbnail_path)
                print(f"Thumbnail generated: {thumbnail_path}")

                # Save event in the database with thumbnail
                event = Event(event_type='Periodic', description='Periodic buffer save',
                            clip=f'event_clips/{video_filename}', thumbnail=f'thumbnails/{thumbnail_filename}')
                event.save()

                # Pass the video file path to the SendEmail instance
                self.send_email.set_video_file_path(video_file_path)
                self.email_executor.submit(self.send_email.send_email_snapshot)
                self.dashboard_api.send_video(video_file_path, description="Periodic buffer save",
                                            thumbnail_path=f'thumbnails/{thumbnail_filename}')

            except subprocess.CalledProcessError as e:
                print(f"Failed to generate thumbnail: {e.stderr.decode()}")
            except Exception as e:
                print(f"Unexpected error during thumbnail generation: {str(e)}")

        # Clear the buffer after saving the clip
        self.running_buffer = []

        # Restart the timer to repeat the process
        self.save_timer = threading.Timer(60, self.save_running_buffer_clip)
        self.save_timer.start()


    def generate_thumbnail(self, video_path, thumbnail_path, time="00:00:00"):
        """
        Generates a thumbnail image from the video at the specified time.

        Args:
            video_path (str): The path to the video file.
            thumbnail_path (str): The path where the thumbnail image will be saved.
            time (str): The time in the video to capture the thumbnail (format: 'HH:MM:SS').
        """
        command = [
            'ffmpeg',
            '-ss', time,  # Time to capture the thumbnail (e.g., 5 seconds into the video)
            '-i', video_path,  # Input video file
            '-vframes', '1',  # Capture just one frame
            '-q:v', '2',  # Output quality level (lower is better quality)
            thumbnail_path  # Output thumbnail file path
        ]
        try:
            result = subprocess.run(command, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            # Capture and print detailed error information
            print(f"FFmpeg command failed with error: {e.stderr.decode()}")
            raise

# Check for file size stabilization
def wait_for_file_stabilization(file_path, timeout=10, interval=0.5):
    start_time = time.time()
    last_size = -1
    while True:
        try:
            current_size = os.path.getsize(file_path)
        except OSError:
            # File does not exist yet
            current_size = -1
        
        if current_size == last_size and current_size > 0:
            # File size has stabilized and it's not empty
            return True
        
        last_size = current_size
        
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout: {file_path} did not stabilize after {timeout} seconds")
        
        time.sleep(interval)  # Wait for a short interval before checking again




