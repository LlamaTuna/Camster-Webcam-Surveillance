# camera/models.py
from django.contrib.auth.models import AbstractUser, Group, Permission, User
from django.db import models
from django.conf import settings
from .encryption import EncryptedCharField

class CustomUser(AbstractUser):
    """
    Custom user model extending Django's AbstractUser. Adds a role field and modifies
    the groups and user_permissions fields to use custom related names and query names.
    
    Attributes:
        ROLE_CHOICES (tuple): Choices for the role field, either 'admin' or 'viewer'.
        role (CharField): Role of the user, with a default value of 'viewer'.
        groups (ManyToManyField): Groups this user belongs to, using a custom related name.
        user_permissions (ManyToManyField): Specific permissions for this user, using a custom related name.
    """
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('viewer', 'Viewer'),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='viewer')

    groups = models.ManyToManyField(
        Group,
        related_name='customuser_set',
        blank=True,
        help_text=('The groups this user belongs to. A user will get all permissions '
                   'granted to each of their groups.'),
        related_query_name='customuser',
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name='customuser_set',
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='customuser',
    )

class Event(models.Model):
    """
    Model representing an event, which includes a timestamp, event type, description,
    an optional video clip associated with the event, and an optional thumbnail image.
    
    Attributes:
        timestamp (DateTimeField): The timestamp when the event was created, automatically set.
        event_type (CharField): The type of event, with a maximum length of 100 characters.
        description (TextField): A detailed description of the event.
        clip (FileField): An optional file field to upload a video clip associated with the event.
        thumbnail (ImageField): An optional image field to upload a thumbnail associated with the event.
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    event_type = models.CharField(max_length=100)
    description = models.TextField()
    clip = models.FileField(upload_to='event_clips/', null=True, blank=True)
    thumbnail = models.ImageField(upload_to='thumbnails/', null=True, blank=True) 

class Face(models.Model):
    """
    Model representing a face, which includes the name, timestamp of when it was recorded,
    the image file, and whether it has been tagged or not.
    
    Attributes:
        name (CharField): The name associated with the face, with a maximum length of 100 characters.
        timestamp (DateTimeField): The timestamp when the face was recorded, automatically set.
        image (ImageField): The image file associated with the face, uploaded to the 'faces_seen/' directory.
        tagged (BooleanField): A boolean indicating whether the face has been tagged, with a default value of False.
        embedding (BinaryField): The 512-dimensional FaceNet embedding stored as binary, for face matching.
    """
    name = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='faces_seen/')
    tagged = models.BooleanField(default=False)
    embedding = models.BinaryField(null=True, blank=True, help_text="512-dim FaceNet embedding")

class EmailSettings(models.Model):
    """
    Model representing the email settings for a user, including SMTP server details.
    
    Attributes:
        user (OneToOneField): A one-to-one relationship with the CustomUser model.
        email (EmailField): The email address associated with the user.
        smtp_server (CharField): The SMTP server address used for sending emails.
        smtp_port (IntegerField): The port number used by the SMTP server.
        smtp_user (CharField): The username used for SMTP authentication.
        smtp_password (CharField): The password used for SMTP authentication.
    """
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    email = models.EmailField()
    smtp_server = models.CharField(max_length=100)
    smtp_port = models.IntegerField()
    smtp_user = models.CharField(max_length=100)
    smtp_password = EncryptedCharField(max_length=500)

class AudioDeviceSetting(models.Model):
    """
    Model representing the audio device settings associated with a user and camera device.
    
    Attributes:
        user (ForeignKey): A foreign key relationship with the CustomUser model.
        device_path (CharField): The path to the camera device, used instead of an index.
        audio_device (CharField): The name of the audio device associated with the camera, defaulting to 'default'.
    """
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    device_path = models.CharField(max_length=255)  # Use device path instead of index
    audio_device = models.CharField(max_length=255, default='default')

    def __str__(self):
        return f"{self.user.username} - {self.device_path} - {self.audio_device}"


class RecognitionSettings(models.Model):
    """
    Model representing face recognition settings.
    
    Attributes:
        similarity_threshold (FloatField): Cosine similarity threshold for face recognition (0.0-1.0).
            Higher values require closer matches. Default is 0.6.
    """
    similarity_threshold = models.FloatField(
        default=0.6,
        help_text="Cosine similarity threshold for face recognition (0.0-1.0). Higher = stricter matching."
    )
    
    class Meta:
        verbose_name = "Recognition Settings"
        verbose_name_plural = "Recognition Settings"
    
    def __str__(self):
        return f"Recognition Settings (threshold: {self.similarity_threshold})"
    
    @classmethod
    def get_settings(cls):
        """Get or create the singleton settings instance."""
        settings_obj, _ = cls.objects.get_or_create(pk=1)
        return settings_obj
