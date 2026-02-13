"""
Provides an EncryptedCharField for Django models that encrypts data at rest
using Fernet symmetric encryption derived from Django's SECRET_KEY.
"""
import base64
import hashlib
from django.conf import settings
from django.db import models
from cryptography.fernet import Fernet, InvalidToken


def _get_fernet():
    """
    Derives a Fernet-compatible key from Django's SECRET_KEY using SHA-256,
    then returns a Fernet instance for encryption/decryption.
    """
    key = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key)
    return Fernet(fernet_key)


def encrypt_value(value):
    """Encrypts a plaintext string and returns a Base64-encoded ciphertext string."""
    if not value:
        return value
    f = _get_fernet()
    return f.encrypt(value.encode()).decode()


def decrypt_value(value):
    """Decrypts a Base64-encoded ciphertext string and returns the plaintext."""
    if not value:
        return value
    f = _get_fernet()
    try:
        return f.decrypt(value.encode()).decode()
    except (InvalidToken, Exception):
        # Value is likely already plaintext (pre-migration data)
        return value


class EncryptedCharField(models.CharField):
    """
    A CharField that transparently encrypts values before saving to the database
    and decrypts them when reading. Uses Fernet symmetric encryption derived
    from Django's SECRET_KEY.
    """

    def get_prep_value(self, value):
        """Encrypt the value before saving to the database."""
        value = super().get_prep_value(value)
        if value is None:
            return value
        # Don't double-encrypt: check if already encrypted
        try:
            f = _get_fernet()
            f.decrypt(value.encode())
            # If decryption succeeds, it's already encrypted
            return value
        except (InvalidToken, Exception):
            # Not encrypted yet, encrypt it
            return encrypt_value(value)

    def from_db_value(self, value, expression, connection):
        """Decrypt the value when reading from the database."""
        return decrypt_value(value)

    def to_python(self, value):
        """Ensure the value is decrypted when accessed as a Python object."""
        value = super().to_python(value)
        if value is None:
            return value
        # Try to decrypt; if it fails, return as-is (plaintext from form input)
        try:
            f = _get_fernet()
            return f.decrypt(value.encode()).decode()
        except (InvalidToken, Exception):
            return value
