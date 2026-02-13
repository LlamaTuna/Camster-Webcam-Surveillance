# Camster: Intelligent Video Surveillance Application

Camster is an advanced video surveillance application that leverages machine learning libraries to intelligently detect movement, recognize faces, and classify objects in a video stream. It is designed to provide real-time monitoring and alerting, making it an ideal solution for security systems, home monitoring, or any other situation requiring automated video surveillance.

## Features

- **Movement Detection:** Detects movement in video frames by comparing the current frame with the previous frame. When movement is detected, the application highlights the movement and logs the event.
- **Facial Recognition:** Uses machine learning to detect and recognize faces in video frames. Recognized faces can be tagged and logged, with the option to alert the user.
- **Object Classification:** Automatically classifies objects detected in video frames using a trained object classifier model. This can be used to identify specific types of objects in the surveillance area.
- **Real-time Alerts:** Sends real-time alerts via email when movement is detected or a known face appears in the video stream. Alerts can include snapshots or video clips of the detected event.
- **Dashboard Integration:** Logs and videos are sent to a connected dashboard API, allowing for remote monitoring and analysis of the surveillance data.
- **Audio Device Management:** Supports selection of audio devices for each camera stream, enabling synchronized audio and video recording.

## Installation

### Prerequisites

- Python 3.12+
- Django 4.2+
- OpenCV 4.x
- PyTorch 2.x (with CUDA support optional)
- facenet-pytorch (for face detection and recognition)
- FFmpeg (for video processing)

### Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/LlamaTuna/Camster-Webcam-Surveillance.git
    ```
2. **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Set up the Django project:**
    ```bash
    python manage.py migrate
    python manage.py createsuperuser
    ```
4. **Start the Django development server:**
    ```bash
    python manage.py runserver
    ```

## Usage

### Adding Cameras

- **Device Enumeration:** The application automatically detects available camera devices on the host system. Users can select cameras to monitor from the list of detected devices.

### Real-time Monitoring

- **Live Video Feed:** Users can view live video feeds from connected cameras through the web interface. The feed is updated in real-time, with detected movement and faces highlighted.
- **Event Logging:** Events such as movement detection, facial recognition, and object classification are logged with timestamps and descriptions. These logs are accessible through the web interface.

### Alerts

- **Email Notifications:** The application can send email notifications when certain events occur (e.g., movement detected, known face recognized). Users can configure email settings directly in the application.

### Video Clips

- **Periodic Buffer Saving:** The application periodically saves video clips from the running buffer, ensuring that important moments are captured. Thumbnails are generated for each video clip, which can be viewed in the application.

## Contributing

Contributions to Camster are welcome! Please fork the repository and submit a pull request with your improvements.

## License

Camster is released under the MIT License.

## Docker Setup and Usage

This section provides instructions on how to build and run the Camster application using Docker.

### Prerequisites

Ensure you have Docker installed on your system. You can follow the [official Docker installation guide](https://docs.docker.com/get-docker/) to set it up.

### Step 1: Build the Docker Image

** To build the Docker image for the Camster application, run the following command in the root directory of the project (where the Dockerfile is located):** 

```bash
docker build -t camster_app:v1.0 .
```


This command will create a Docker image named camster_app with the tag v1.0. The build process might take a few minutes as it installs the necessary dependencies and copies the required files into the container.

### Step 2: Run the Docker Container
Once the Docker image is built, you can run the container using the following command:
```bash
docker run -p 80:80 camster_app:v1.0
```

This command will start the application inside a Docker container and map port 80 of the container to port 80 of your host machine. You can access the application by navigating to http://localhost in your web browser.

### Step 3: Accessing the Application
With the container running, you can access the Camster web application by opening a web browser and going to:
```bash
http://localhost
```

If you want to stop the container, you can do so by identifying the container ID and stopping it with the following commands:

List running containers:

```bash
docker ps
```

Stop the container:

```bash
docker stop <container_id>
```

Replace <container_id> with the actual container ID from the output of the docker ps command.

### Troubleshooting
Port Conflicts: If port 80 is already in use on your machine, you can map the container to a different port using the -p option, for example -p 8080:80, and then access the application at http://localhost:8080.
File Changes: If you make changes to the application code, you’ll need to rebuild the Docker image using the docker build command mentioned above.

### Cleaning Up
To remove the Docker image and free up space on your system, you can use the following commands:

Remove the container (if still running):

```bash
docker rm <container_id>
```


Remove the Docker image:

```bash
docker rmi camster_app:v1.0
```