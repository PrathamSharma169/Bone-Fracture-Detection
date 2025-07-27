# Bone Fracture Classification and Quality Evaluation Using Image Analysis

This project implements a convolutional neural network (CNN) for classifying bone fractures using image analysis. The model is trained on a dataset of X-ray images and can be deployed as a web application using Flask.

## Project Structure

```
bone-fracture-classification
├── src
│   ├── training.py       # Script to train the CNN model
│   ├── app.py            # Flask application for image upload and prediction
│   └── templates
│       └── index.html    # HTML template for the web application
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd bone-fracture-classification
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   Run the `training.py` script to train the CNN model and save it as `model.pkl`.
   ```bash
   python src/training.py
   ```

4. **Run the Flask application**:
   Start the Flask server by running the `app.py` script.
   ```bash
   python src/app.py
   ```

5. **Access the web application**:
   Open your web browser and go to `http://127.0.0.1:5000` to access the application. You can upload an image and get predictions on whether the bone is fractured or not.

## Usage

- Upload an X-ray image of a bone through the web interface.
- The model will process the image and display the prediction results.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.