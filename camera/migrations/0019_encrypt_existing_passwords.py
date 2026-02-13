"""
Data migration to encrypt existing plaintext SMTP passwords.
"""
from django.db import migrations


def encrypt_existing_passwords(apps, schema_editor):
    """Encrypt any existing plaintext SMTP passwords using Fernet."""
    import base64
    import hashlib
    from django.conf import settings as django_settings
    from cryptography.fernet import Fernet, InvalidToken

    # Derive the Fernet key from SECRET_KEY
    key = hashlib.sha256(django_settings.SECRET_KEY.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key)
    f = Fernet(fernet_key)

    EmailSettings = apps.get_model('camera', 'EmailSettings')
    for es in EmailSettings.objects.all():
        if es.smtp_password:
            # Check if already encrypted
            try:
                f.decrypt(es.smtp_password.encode())
                # Already encrypted, skip
                continue
            except (InvalidToken, Exception):
                pass
            # Encrypt the plaintext password
            es.smtp_password = f.encrypt(es.smtp_password.encode()).decode()
            es.save(update_fields=['smtp_password'])


def decrypt_existing_passwords(apps, schema_editor):
    """Reverse migration: decrypt SMTP passwords back to plaintext."""
    import base64
    import hashlib
    from django.conf import settings as django_settings
    from cryptography.fernet import Fernet, InvalidToken

    key = hashlib.sha256(django_settings.SECRET_KEY.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key)
    f = Fernet(fernet_key)

    EmailSettings = apps.get_model('camera', 'EmailSettings')
    for es in EmailSettings.objects.all():
        if es.smtp_password:
            try:
                es.smtp_password = f.decrypt(es.smtp_password.encode()).decode()
                es.save(update_fields=['smtp_password'])
            except (InvalidToken, Exception):
                pass  # Already plaintext


class Migration(migrations.Migration):

    dependencies = [
        ('camera', '0018_encrypt_smtp_password'),
    ]

    operations = [
        migrations.RunPython(
            encrypt_existing_passwords,
            reverse_code=decrypt_existing_passwords,
        ),
    ]
