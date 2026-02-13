"""
Management command to clean up old untagged face captures.

Usage:
    python manage.py cleanup_faces              # Delete untagged faces older than 7 days
    python manage.py cleanup_faces --days 3     # Delete untagged faces older than 3 days
    python manage.py cleanup_faces --dry-run    # Preview what would be deleted
"""
import os
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.utils import timezone
from camera.models import Face


class Command(BaseCommand):
    help = 'Delete untagged face captures older than the specified number of days'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Delete untagged faces older than this many days (default: 7)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Preview what would be deleted without actually deleting',
        )

    def handle(self, *args, **options):
        days = options['days']
        dry_run = options['dry_run']
        cutoff = timezone.now() - timedelta(days=days)

        # Find untagged faces older than cutoff
        old_faces = Face.objects.filter(tagged=False, timestamp__lt=cutoff)
        count = old_faces.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS(
                f'No untagged faces older than {days} days found.'
            ))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING(
                f'[DRY RUN] Would delete {count} untagged face(s) older than {days} days:'
            ))
            for face in old_faces[:20]:  # Show first 20
                self.stdout.write(f'  - {face.name} ({face.timestamp:%Y-%m-%d %H:%M}) {face.image}')
            if count > 20:
                self.stdout.write(f'  ... and {count - 20} more')
            return

        # Delete image files from disk
        deleted_files = 0
        for face in old_faces:
            if face.image:
                try:
                    image_path = face.image.path
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        deleted_files += 1
                except Exception as e:
                    self.stderr.write(f'Error deleting file for {face.name}: {e}')

        # Delete database records
        old_faces.delete()

        self.stdout.write(self.style.SUCCESS(
            f'Deleted {count} untagged face record(s) and {deleted_files} image file(s) '
            f'older than {days} days.'
        ))
