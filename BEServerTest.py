import unittest
import random
from collections import deque
from preprocessing import collect_test_data, calculate_majority_vote
import pdb
import json
import numpy as np
import joblib
# Assuming the functions and variables from the script are imported here

# class TestCollectTestData():  #unittest.TestCase

    # def test_data_collection_and_processing():
    #     # Send data to the test function
    #     tests_data = [
    #         ["C=-64=27", "A=-77=82", "B=-60=25"],
    #         # ["C=-66=96", "A=-80=9", "B=-57=35"],
    #         # ["C=-71=46", "A=-85=83", "B=-67=38"],
    #         # ["C=-71=81", "A=-80=22", "B=-65=55"],
    #         # ["C=-68=75", "A=-79=60", "B=-65=20"],
    #         # ["C=-72=82", "A=-73=21", "B=-59=87"],
    #         # ["C=-63=88", "A=-82=92", "B=-54=87"],
    #         # ["C=-70=65", "A=-75=58", "B=-54=73"],
    #         # ["C=-73=44", "A=-72=81", "B=-58=98"],
    #         # ["C=-70=97", "A=-82=29", "B=-72=7"]
    #     ]


    #     for test_data in tests_data:
    #         for data in test_data:
    #             collect_test_data(data)




def calculate_majority_vote():
    history = deque()
    feature_data = [[0, 0, 0], [-82, -72, -70], [-70,-82,-72]]
    for f in feature_data: 
        rf_model = joblib.load('random_forest_model.pkl')
        knn_model = joblib.load('knn_model.pkl')
        feature_data_array = np.array(f).reshape(1, -1)
        rf_prediction = rf_model.predict(feature_data_array)
        knn_prediction = knn_model.predict(feature_data_array)
        history.append((rf_prediction, knn_prediction))  
        print(history)
        for predictions in history:
            prediction = predictions[0]
            prediction_key = prediction.item() if isinstance(prediction, np.ndarray) else prediction
            print(prediction_key)

calculate_majority_vote()
    


# if __name__ == "__main__":

    # BE
