ObjectDetection-RaspberryPi

This project enables your Raspberry Pi to detect objects in real-time using TensorFlow Lite and a webcam. It identifies objects, draws bounding boxes around them, and labels them on the screen.
Requirements

    Raspberry Pi (or any device with a webcam)
    TensorFlow Lite
    OpenCV
    Python 3.7.3 (For this code to work, as the default Python version on Raspberry Pi might be 3.11.2)
    Virtual Environment (Optional): If you're using Python 3.11.2, creating a virtual environment is recommended.

Features

    Real-Time Object Detection: Detects objects in the webcam feed using the TensorFlow Lite model.
    Bounding Boxes and Labels: Highlights detected objects with bounding boxes and labels.
    FPS Display: Shows the frames per second (FPS) for performance monitoring.

Installation

    Clone the TensorFlow Lite repository and install dependencies:

    First, create a new directory, then clone the TensorFlow Lite examples repository and navigate to the object detection folder:

cd <new_directory>
git clone https://github.com/tensorflow/examples --depth 1
cd examples/lite/examples/object_detection/raspberry_pi/
sh setup.sh
pip install -r requirements.txt

Set up Python 3.7.3 and virtual environment (Optional but recommended):

If your Raspberry Pi has Python 3.11.2, use Python 3.7.3 by following these steps:

sudo apt install python3.7 python3.7-venv python3.7-dev
python3.7 -m venv ~/tflite1-env
source ~/tflite1-env/bin/activate

Then run the detection script:

python detect.py

Once done, deactivate the virtual environment:

deactivate

Run the object detection script:

    Connect a webcam to your Raspberry Pi.
    Run the Python script to start real-time object detection:

    python object_detection.py

    Exit the application: Press q to exit the application.

How It Works

    The program uses a pre-trained TensorFlow Lite model (efficientdet_lite0.tflite) to detect objects in the webcam feed.
    Detected objects are highlighted with bounding boxes and labeled with their names.
    The program calculates and displays the frames per second (FPS) to monitor the performance
