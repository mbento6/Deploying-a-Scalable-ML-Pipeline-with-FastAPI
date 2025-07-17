# Deploying a Scalable ML Pipeline with FastAPI

This project demonstrates how to build, train, and deploy a machine learning model using a complete MLOps pipeline. The model is trained to predict whether an individual earns more than $50,000 per year using demographic data from the census dataset. The project includes data processing, model training and evaluation, RESTful API deployment using FastAPI, and automated testing and integration using GitHub Actions.

GitHub repository: https://github.com/mbento6/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git

## Features

- Data cleaning and feature engineering
- Model training using RandomForestClassifier
- Evaluation metrics: precision, recall, and F1 score
- Slice-based performance metrics for fairness analysis
- FastAPI-based RESTful service for model inference
- GET and POST endpoints
- Input validation using Pydantic
- Unit tests using pytest
- Continuous integration using GitHub Actions

## File Structure

├── data/ Raw input data
├── model/ Trained model, encoder, and label binarizer
├── ml/
│ ├── data.py Functions for processing data
│ ├── model.py Functions for training and inference
├── main.py FastAPI application
├── local_api.py Sends GET and POST requests for testing
├── slice_output.txt Slice-based model performance results
├── test_ml.py Unit tests
├── model_card.md Model documentation
├── screenshots/
│ └── continuous_integration.png
├── .github/workflows/
│ └── main.yml CI configuration
├── requirements.txt Project dependencies
├── README.md Project overview


## Running the Application Locally

1. Clone the repository:

git clone https://github.com/mbento6/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI

2. Create and activate a virtual environment:

conda create -n fastapi python=3.10 -y
conda activate fastapi
pip install -r requirements.txt


3. Train the model and save artifacts:

python train_model.py


4. Start the FastAPI server:

uvicorn main:app --reload

Visit the Swagger UI at http://127.0.0.1:8000/docs to interact with the API.

5. Test the API locally:

python local_api.py

## Running Tests

To run the unit tests:

pytest test_ml.py -v


## Continuous Integration

This project uses GitHub Actions to perform automated testing and linting. A screenshot showing the passing workflow is included at:

`screenshots/continuous_integration.png`

## Submission Details

Public GitHub repository: https://github.com/mbento6/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git  
The GitHub repository link is also included in this README and should be provided in the "Submission Details" section upon submission.

## Author

Mike Benton  
Email: mbento6@wgu.edu

## License

This project is licensed for educational purposes only.
