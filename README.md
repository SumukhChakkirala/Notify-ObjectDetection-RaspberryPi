# ObjectDetection-RaspberryPi
This project allows your Raspberry Pi to detect objects in real-time using TensorFlow Lite and a webcam. It identifies objects, draws bounding boxes around them, and labels them on the screen.

we need python 3.7.3 for the code to work. my raspberry pi had a default python3.11.2 version, so i had to use a virtual environment 

## Features

- Real-Time Object Detection: Detects objects in the webcam feed using the TensorFlow Lite model.
- Bounding Boxes and Labels: Highlights detected objects with bounding boxes and labels.
- FPS Display: Shows the frames per second (FPS) for performance monitoring.

## Requirements

- Raspberry Pi (or any device with a webcam)
- TensorFlow Lite
- OpenCV
- Python 3.7.3

## Installation

- Install dependencies:
    1. make a new directory and clone TensorFlow Lite repo:
    ```
    cd <new_directory>
    git clone https://github.com/tensorflow/examples --depth 1
    cd examples/
    cd lite/
    cd examples/
    cd object_detection/
    cd raspberry_pi/
    sh setup.sh
    pip install -r requirements.txt
    ```

- For the virtual environment(optional):
    ```
    sudo apt install python3.7 python3.7-venv python3.7-dev
    python3.7 -m venv ~/tflite1-env
    source ~/tflite1-env/bin/activate
    python detect.py
    deactivate
    ```

## Usage

- Connect a webcam to your Raspberry Pi.
- Run the Python script:
    ```
    python detect.py
    ```
    The program will start capturing video from the webcam and perform real-time object detection.
- Press q to exit the application.

## How It Works

- The program uses a pre-trained TensorFlow Lite model (efficientdet_lite0.tflite) to detect objects in the webcam feed.
- Detected objects are highlighted with bounding boxes and labeled with their names.
- The program calculates and displays the frames per second (FPS) to monitor the performance.
