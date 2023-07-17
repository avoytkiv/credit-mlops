from pathlib import Path
import uvicorn
from fastapi import FastAPI
import pandas as pd
from joblib import load
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

# Defining the path to the model
MODEL_JOBLIB_PATH = Path('models/model.pkl').absolute()
PCA_JOBLIB_PATH = Path('models/featurizer.pkl').absolute()

# Creating the App object
app = FastAPI()

# Using Pydantic lib, defining the data type for all the inputs from iris dataset
class ModelInput(BaseModel):
    '''
    Defining the data type for all the inputs from processed dataset
    '''
    BALANCE: float
    BALANCE_FREQUENCY: float
    PURCHASES: float
    ONEOFF_PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    PURCHASES_FREQUENCY: float
    ONEOFF_PURCHASES_FREQUENCY: float
    PURCHASES_INSTALLMENTS_FREQUENCY: float
    CASH_ADVANCE_FREQUENCY: float
    CASH_ADVANCE_TRX: float
    PURCHASES_TRX: float
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    PRC_FULL_PAYMENT: float
    TENURE: float

# Load the model from disk
model = load(MODEL_JOBLIB_PATH)
pca = load(PCA_JOBLIB_PATH)


# Defining the prediction function
def predict(data):
    '''
    Defining the prediction function
    '''
    data_df = pd.DataFrame([data])  # Convert input dictionary to DataFrame
    data_featurized = pca.transform(StandardScaler().fit_transform(data_df)) # Featurize data
    prediction = model.predict(data_featurized)[0]
    return prediction



# Defining the root endpoint of the API
@app.get('/')
def index():
    '''
    Defining the root endpoint of the API
    '''
    return {'message': 'Hello World'}

# Defining the prediction endpoint of the API
@app.post('/predict')
def get_prediction(data: ModelInput):
    '''
    Defining the prediction endpoint of the API
    '''
    data = data.dict()
    prediction = predict(data)  # Calls the predict function
    return str(prediction)