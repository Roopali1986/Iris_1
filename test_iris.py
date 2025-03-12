import pandas as pd
import pickle
import os.path
from sklearn.metrics import accuracy_score

def test_model_exists():
    assert os.path.isfile('decision_tree_model.pkl'), "Model file doesn't exist"

def test_model_accuracy():
    # This test will run only if the model file exists
    if os.path.isfile('decision_tree_model.pkl'):
        # Load the model
        with open('decision_tree_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load data
        iris = pd.read_csv('Iris.csv')
        
        # Prepare inputs and target
        inputs = iris.drop(['Id', 'Species'], axis=1)
        target = iris['Species']
        
        # Encode target variable consistently with training
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoded_target = encoder.fit_transform(target)
        
        # Predict
        predictions = model.predict(inputs)
        
        # Check accuracy
        accuracy = accuracy_score(encoded_target, predictions)
        assert accuracy > 0.9, f"Model accuracy {accuracy} is below threshold"