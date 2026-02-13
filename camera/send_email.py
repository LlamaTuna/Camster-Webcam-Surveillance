import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import cv2
from datetime import datetime
from .models import EmailSettings
import os

class SendEmail:
    """
    A class responsible for handling the sending of email notifications, including
    attaching snapshots, detected faces, and video clips.
    """

    def __init__(self, request):
        """
        Initializes the SendEmail class with the user's request, setting up buffers
        for alerts, frames, and detected faces.

        Args:
            request: The Django request object containing the user information.
        """
        self.request = request
        self.alert_buffer = []
        self.frame_buffer = []
        self.detected_faces = []
        self.video_file_path = None  # Attribute to hold the video file path

    def log_event(self, event):
        """
        Logs an event into the alert buffer with a timestamp.

        Args:
            event (str): The event description to be logged.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {event}"
        self.alert_buffer.append(log_entry)
        print("SendEmail logged event:", log_entry)

    def set_detected_faces(self, faces):
        """
        Sets the list of detected faces to be included in the email.

        Args:
            faces (list): A list of detected faces with associated labels.
        """
        self.detected_faces = faces

    def set_video_file_path(self, file_path):
        """
        Sets the path of the video file to be attached to the email.

        Args:
            file_path (str): The file path of the video to attach.
        """
        self.video_file_path = file_path

    def send_email_snapshot(self):
        """
        Sends an email with the logged events, detected faces, and optionally attached
        snapshots and video clips. Handles all aspects of email composition and sending.
        """
        print("Attempting to send email snapshot...")
        if not self.alert_buffer:
            print("Alert buffer is empty, no email will be sent.")
            return
        try:
            if self.request and hasattr(self.request, 'user') and self.request.user.is_authenticated:
                email_settings = EmailSettings.objects.get(user=self.request.user)
            else:
                # Fallback: use the first available EmailSettings when user is not authenticated
                email_settings = EmailSettings.objects.first()
                if not email_settings:
                    print("No email settings configured. Skipping email.")
                    return

            print(f"Email Settings: {email_settings.__dict__}")  # Debug statement

            smtp_server = email_settings.smtp_server
            smtp_port = email_settings.smtp_port
            smtp_user = email_settings.smtp_user
            smtp_password = email_settings.smtp_password
            from_email = smtp_user
            to_email = email_settings.email
            subject = "Motion Detection Alert Snapshot"
            body = "\n".join(self.alert_buffer)
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            if self.detected_faces:
                body += "\n\nDetected Faces:\n"
                for i, face in enumerate(self.detected_faces):
                    label = face.get('label', 'Unknown')
                    body += f"Person {i + 1}: {label}\n"

            msg.attach(MIMEText(body, 'plain'))

            # Ensure at least two frames are available
            if len(self.frame_buffer) < 2:
                if len(self.frame_buffer) == 1:
                    # Duplicate the single available frame
                    self.frame_buffer.append(self.frame_buffer[0])
                elif len(self.frame_buffer) == 0:
                    # Handle case with no frames (this should be rare)
                    print("No frames available in frame_buffer, cannot attach images to email.")
                    return

            selected_frames = self.select_representative_frames(self.frame_buffer, 2)

            for i, frame in enumerate(selected_frames):
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_data = img_encoded.tobytes()
                image = MIMEImage(image_data, name=f"event_{i + 1}.jpg")
                msg.attach(image)

            if self.video_file_path:
                with open(self.video_file_path, 'rb') as video_file:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(video_file.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(self.video_file_path)}')
                    msg.attach(part)

            print("Connecting to SMTP server...")
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            print("Logging into SMTP server...")
            server.login(smtp_user, smtp_password)
            text = msg.as_string()
            print("Sending email...")
            server.sendmail(from_email, to_email, text)
            server.quit()

            self.alert_buffer = []
            self.frame_buffer = []
            self.video_file_path = None  # Reset the video file path after sending the email
            print("Email sent successfully")
        except Exception as e:
            print(f"Failed to send snapshot email: {str(e)}")

    def select_representative_frames(self, frames, num_frames):
        """
        Selects a specified number of representative frames from the buffer.

        Args:
            frames (list): A list of frames to select from.
            num_frames (int): The number of frames to select.

        Returns:
            list: A list of selected representative frames.
        """
        if len(frames) <= num_frames:
            return frames
        interval = len(frames) // num_frames
        selected_frames = [frames[i * interval] for i in range(num_frames)]
        return selected_frames
