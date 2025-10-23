# Computer Vision & ML Screw Counter

This project is a complete end-to-end proof-of-concept for a system that counts screws in an image. It leverages a classical computer vision pipeline for object segmentation and a machine learning model for classification, built from scratch without relying on pre-trained object detection models like YOLO.

This project was developed as a technical demonstration of skills in computer vision, machine learning, and software engineering.

---

## Key Features

* **Object Segmentation:** Implements a good pipeline to isolate objects even in challenging conditions.
* **Intelligent Object Separation:** Utilizes the Watershed algorithm to correctly separate touching or overlapping screws.
* **Custom Machine Learning Model:** A `RandomForestClassifier` is trained on custom-labeled data to distinguish between "screws" and "non-screws" (e.g., noise, logos, segmentation artifacts).
* **Interactive Data Collection Tool:** This is the part where I actually spent the most time on: A custom-built tool (`1_data_collector.py`) with a GUI for efficient and high-quality data labeling, allowing for real-time parameter tuning to ensure optimal segmentation for each training image. Since I was trying to do this from scratch, I wasn't planning on using labelimg or other software that could go in the opposite direction of the project, so I did it my own.

---

## Tech Stack

* **Python 3**
* **OpenCV:** For all image processing tasks.
* **Scikit-learn:** For training and evaluating the RandomForest classification model.
* **Scikit-image & SciPy:** For implementing the Watershed algorithm.
* **Pandas:** For data manipulation and handling the features dataset.
* **Numpy:** For numerical operations.
* **Joblib:** For saving and loading the trained model.

---

## Installation & Usage

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.8+
* pip

### Installation

1.  Clone the repo:
    ```sh
    git clone (https://github.com/KauanNogueira/screwcounter.git)
    ```
2.  Navigate to the project directory:
    ```sh
    cd screw_counter
    ```
3.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

The project is divided into three main scripts, intended to be run in order:

1.  **Collect Data (Optional - a sample `treino_classificado.csv` is provided):**
    The interactive tool allows you to create your own dataset. For each image in `dataset/train`, you can adjust segmentation parameters for optimal quality and then label the resulting objects.

    This was one of the things I would like to improve in the future, since it's the main reason the results are not quite aligned with the reality
    ```sh
    python data_collector.py
    ```

2.  **Train the Model:**
    This script reads the `treino_classificado.csv`, trains the model, evaluates its performance, and saves the trained model to `screw_classifier_model.pkl`.
    ```sh
    python train_model.py
    ```

3.  **Run the Counter:**
    This is the final application. It loads an image from the `dataset/test` folder and the trained model to perform the final count.
    ```sh
    python screw_counter.py
    ```
    *Note: Remember to change the `IMAGE_TO_TEST` variable inside the script to your desired test image.*

---

## Challenges & Future Work

This project successfully demonstrates a functional proof-of-concept. The main challenge lies in the robustness of the image segmentation across widely varying image conditions (lighting, backgrounds, object types).

Future improvements could include:

* **Dataset Expansion:** The most impactful next step is to significantly scale up the data collection process. A larger and more diverse dataset would allow the model to generalize much more effectively, making the system more robust to segmentation imperfections.
* **Advanced Feature Engineering:** Experimenting with more advanced image descriptors (e.g., Hu Moments, HOG features) could further improve the model's classification accuracy.
* **Automated Parameter Tuning:** Developing a method to automatically select the best segmentation parameters for a given image would reduce the reliance on a fixed set of "default" parameters.