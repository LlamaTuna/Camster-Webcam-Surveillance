# Generated by Django 4.2 on 2024-07-22 00:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('camera', '0003_alter_emailsettings_smtp_password_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='emailsettings',
            name='smtp_password',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='emailsettings',
            name='smtp_server',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='emailsettings',
            name='smtp_user',
            field=models.CharField(max_length=100),
        ),
    ]
