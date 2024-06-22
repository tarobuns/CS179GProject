# Toyota Car Image Classification System

## Overview
This project focuses on developing a machine learning model to classify images of 38 different Toyota car models. The system involves data collection, preprocessing, feature extraction, model training, and a user-friendly web interface for image classification.

## Highlights
- **Machine Learning**: Utilized PySpark for data processing and HDFS for handling large datasets.
- **Model**: Implemented feature extraction using InceptionV3 and logistic regression for classification.
- **Deployment**: Created a Flask web application for users to upload car images and receive model predictions.
- **Data Management**: Stored model training results and performance metrics in a MySQL database.

## Project Structure
### Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/tarobuns/CS179GProject.git
    cd CS179GProject
    ```
2. **Set up a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

### Usage
1. **Run the web interface**:
    ```sh
    python app.py
    ```
2. **Upload Images**:
    - Open your web browser and navigate to `http://127.0.0.1:5000`.
    - Upload an image of a Toyota car to receive model predictions.

## Technical Details
### Model Training
- **Feature Extraction**: Utilized the InceptionV3 model to extract features from car images.
- **Classification**: Trained a logistic regression model to classify car models based on extracted features.
- **Data Handling**: Employed HDFS for data storage and PySpark for processing.

### Web Interface
- **Framework**: Built with Flask.
- **Functionality**: Allows users to upload images and get predictions.
- **Deployment**: Hosted locally for testing purposes.

## Key Technologies
- **Programming Languages**: Python, SQL
- **Libraries & Frameworks**: PySpark, TensorFlow (InceptionV3), Flask, MySQL
- **Tools**: HDFS, EC2

## Links
- **GitHub Repository**: [CS179GProject](https://github.com/tarobuns/CS179GProject)
