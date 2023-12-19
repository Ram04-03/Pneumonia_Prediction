## Pneumonia Prediction using Tensorflow

This repository contains the code and resources for a web application that uses a Convolutional Neural Network (CNN) model built with TensorFlow to predict pneumonia from chest X-ray images. The user interface is built using Streamlit, making it easy to access and use.

### Getting Started

1. **Prerequisites:**
    * Python 3.7+
    * TensorFlow
    * Streamlit
    * Additional libraries listed in `requirements.txt`
      
2. **Installation:**
    * Clone this repository (`git clone https://github.com/Ram04-03/Pneumonia_Prediction`)
    * Install required libraries (`pip install -r requirements.txt`)
      
3. **Run the web app:**
    * Run `streamlit run app.py` in the terminal

### Using the App

1. Visit http://localhost:8501 in your web browser.
2. Click the "Upload Image" button and select a chest X-ray image.
3. Click the "Predict" button.
4. The app will display the predicted probability of pneumonia and the classification ("Normal" or "Pneumonia").

### Overview

**Dataset:**

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.
Link to the Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Model:**

* The CNN model is trained on a dataset of chest X-ray images labeled as either normal or pneumonia.
* The model architecture and training details are available in the `pneumonia-prediction.ipynb` file.
* You can train the model on your own dataset or fine-tune the existing model by modifying the code.

**Web App:**

* The app built using Streamlit provides a user-friendly interface for interacting with the model.
* The code for the app is located in `webappapp.py`.

**Results:**
* The model has built with 94.54% Train accuracy and 90.38% Test accuracy
* The web app has been built and hosted in streamlit
* The link for the webapp which has been built --> https://pneumonia-pred.streamlit.app/
