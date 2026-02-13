# urls.py
from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views
from .views import log_event, device_settings

urlpatterns = [
    path('admin/', views.admin_view, name='admin_view'),
    path('', views.index, name='index'),
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='camera/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('list_faces/', views.list_faces, name='list_faces'),
    path('tag_face/<int:face_id>/', views.tag_face, name='tag_face'),
    path('video_feed/<path:device_path>/', views.video_feed, name='video_feed'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('get_logs/', views.get_logs, name='get_logs'),
    path('upload_face/', views.upload_face, name='upload_face'),
    path('email_settings/', views.email_settings, name='email_settings'),
    path('user_settings/', views.user_settings, name='user_settings'),
    path('recognition_settings/', views.recognition_settings, name='recognition_settings'),
    path('delete_all_faces/', views.delete_all_faces, name='delete_all_faces'),
    path('api/log_event/', log_event, name='log_event'),
    path('device-settings/', device_settings, name='device_settings'),

]

"""
URL patterns for the camera app.

Each URL pattern maps a specific URL path to a view function or class-based view.
"""

# Explanation of URL patterns:

"""
admin/:
    Purpose: Displays the admin view.
    View Function: views.admin_view
    Name: admin_view
"""

"""
/:
    Purpose: Displays the index page.
    View Function: views.index
    Name: index
"""

"""
register/:
    Purpose: Allows users to register.
    View Function: views.register
    Name: register
"""

"""
login/:
    Purpose: Allows users to log in.
    View Class: auth_views.LoginView
    Template: camera/login.html
    Name: login
"""

"""
logout/:
    Purpose: Allows users to log out.
    View Class: auth_views.LogoutView
    Name: logout
"""

"""
list_faces/:
    Purpose: Displays a list of faces.
    View Function: views.list_faces
    Name: list_faces
"""

"""
tag_face/<int:face_id>/:
    Purpose: Allows users to tag a face with a given ID.
    View Function: views.tag_face
    Name: tag_face
"""

"""
video_feed/:
    Purpose: Streams the video feed.
    View Function: views.video_feed
    Name: video_feed
"""

"""
accounts/:
    Purpose: Includes the default authentication URLs provided by Django.
    Include: django.contrib.auth.urls
"""

"""
get_logs/:
    Purpose: Retrieves the logs.
    View Function: views.get_logs
    Name: get_logs
"""

"""
upload_face/:
    Purpose: Allows users to upload a face.
    View Function: views.upload_face
    Name: upload_face
"""
