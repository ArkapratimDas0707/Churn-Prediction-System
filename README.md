Churn Prediction API

This is a production-ready machine learning API that predicts customer churn using the Telco dataset. It includes data preprocessing, model training, and deployment using Flask and Docker. The API is deployed via Render and supports CI/CD via GitHub Actions.

## Features

- Trained with a Random Forest model
- Modular codebase (preprocessing, modeling, deployment separated)
- Dockerized for easy deployment
- REST API built with Flask
- CI/CD pipeline using GitHub Actions

## Project Structure

├── app/ # Flask API │ └── app.py ├── models/ # Trained models (.pkl) ├── scripts/ # Data preprocessing scripts │ └── preprocess.py ├── src/ # Supporting modules ├── .dockerignore ├── Dockerfile ├── requirements.txt ├── .github/workflows/ # CI/CD configs │ └── docker.yml ├── test_request.py # Test script for sending sample input └── README.md


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ArkapratimDas0707/Telco_Churn.git
cd Telco_Churn


# Create virtual environment

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Run locally
python app/app.py
Visit: http://127.0.0.1:5000/

5. Test the API
Use the test_request.py to send a POST request to /predict with sample input.

docker build -t telco-churn-api .
docker run -p 5000:5000 telco-churn-api


API Endpoint
POST /predict
Accepts customer information and returns a churn prediction.

Sample Request:

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  ...
}

Response:

{
  "prediction": "Churn"
}


CI/CD
This project uses GitHub Actions to:

Install dependencies

Run linting or tests

Build Docker image

Workflow file: .github/workflows/docker.yml

License
This project is open-source and available for educational and non-commercial use.


---

Let me know if you'd like a version that includes test instructions or an example JSON body. Or we can auto-generate one from your feature columns!





