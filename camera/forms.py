from django import forms
from .models import Face
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import CustomUser, EmailSettings, User, AudioDeviceSetting, RecognitionSettings
import sounddevice as sd

class CustomUserCreationForm(UserCreationForm):
    """
    A form that creates a user, with no privileges, from the given username, email, and password.
    """
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2', 'role']

class CustomAuthenticationForm(AuthenticationForm):
    """
    A form used for user authentication.
    """
    pass

class TagFaceForm(forms.ModelForm):
    """
    A form for tagging faces with a name and an image.
    """
    class Meta:
        model = Face
        fields = ['name', 'image']

class RegisterForm(forms.ModelForm):
    """
    A form for registering a new user, including password confirmation.
    """
    password = forms.CharField(widget=forms.PasswordInput)
    password2 = forms.CharField(widget=forms.PasswordInput, label='Confirm password')

    class Meta:
        model = User
        fields = ['username', 'email']

    def clean_password2(self):
        """
        Verifies that the entered passwords match.
        """
        cd = self.cleaned_data
        if cd['password'] != cd['password2']:
            raise forms.ValidationError('Passwords don\'t match.')
        return cd['password2']
    
class UploadFaceForm(forms.ModelForm):
    """
    A form for uploading a face image with a name.
    """
    class Meta:
        model = Face
        fields = ['name', 'image']

class EmailSettingsForm(forms.ModelForm):
    """
    A form for updating the email settings of a user, including SMTP server details.
    """
    class Meta:
        model = EmailSettings
        fields = ['smtp_server', 'smtp_port', 'smtp_user', 'smtp_password', 'email']
        widgets = {
            'smtp_password': forms.PasswordInput(attrs={'placeholder': '●●●●●●●●'}),
        }

class UserSettingsForm(forms.ModelForm):
    """
    A form for updating user settings, such as first name, last name, and email.
    """
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

class AudioDeviceSettingForm(forms.ModelForm):
    """
    A form for configuring the audio device settings associated with a camera.
    
    Attributes:
        camera_index (CharField): Hidden input field for storing the camera index.
    """
    camera_index = forms.CharField(widget=forms.HiddenInput())

    def __init__(self, *args, **kwargs):
        """
        Initializes the form and sets up the available audio devices as choices.
        """
        super(AudioDeviceSettingForm, self).__init__(*args, **kwargs)
        devices = sd.query_devices()
        input_devices = [(str(index), device['name']) for index, device in enumerate(devices) if device['max_input_channels'] > 0]
        print("Form input devices:", input_devices)
        self.fields['audio_device'] = forms.ChoiceField(choices=input_devices)

    class Meta:
        model = AudioDeviceSetting
        fields = ['camera_index', 'audio_device']


class RecognitionSettingsForm(forms.ModelForm):
    """
    A form for configuring face recognition settings, including similarity threshold.
    """
    similarity_threshold = forms.FloatField(
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'step': '0.05',
            'class': 'form-control',
            'style': 'width: 200px;'
        }),
        help_text='Cosine similarity threshold (0.0-1.0). Higher = stricter matching. Recommended: 0.5-0.7'
    )
    
    class Meta:
        model = RecognitionSettings
        fields = ['similarity_threshold']