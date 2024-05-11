import csv
import joblib
from collections import deque
import json
import numpy as np

#Prepeocessing

# result = [[-1, "A", "B", "C"]] + [[i, None, None, None] for i in range(1, 32)]
result_A = []
result_B = []
result_C = []
num_data_points = 5

# History of predictions
history = deque(maxlen=20)

train = False

# Load the models and their accuracies
rf_model = joblib.load('random_forest_model.pkl')
knn_model = joblib.load('knn_model.pkl')
rf_accuracy = 90.0  # Hardcoded from your provided accuracy
knn_accuracy = 96.67  # Hardcoded from your provided accuracy

def add_data_point(data_point):
    # Split the data point using '='
    key, value, data_number = data_point.split("=")
    # Write to result array
    print("key: ", key)
    print("value: ", value)
    if key == 'A':
        result_A.append(value)
    elif key == 'B':
        result_B.append(value)
    elif key == 'C':
        result_C.append(value)
    return

def write_to_file():
    print("Writing to file...")
    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for ele in zip(result_A, result_B, result_C):
            writer.writerow(ele)
    result_A.clear()
    result_B.clear()
    result_C.clear()
    return True

def collect_data(data):
    add_data_point(data)
    if len(result_A) >= num_data_points and len(result_B) >= num_data_points and len(result_C) >= num_data_points:
        write_to_file()
    
def collect_test_data(data):
    add_data_point(data)
    global result_A, result_B, result_C, history
    print('result_A ' + str(len(result_A)) + str(result_A))
    print('result_B ' + str(len(result_B)) + str(result_B))
    print('result_C ' + str(len(result_C)) + str(result_C))
    #only batch this when the next set of results come in
    if len(result_A) >= 1 and len(result_B) >= 1 and len(result_C) >= 1:
        print(result_A[0], result_B[0], result_C[0])
        feature_data = [result_A[len(result_A) - 1], result_B[len(result_B) - 1], result_C[len(result_C) - 1]]
        feature_data_array = np.array(feature_data).reshape(1, -1)
        rf_prediction = rf_model.predict(feature_data_array)
        knn_prediction = knn_model.predict(feature_data_array)
        history.append((rf_prediction, knn_prediction))  

        room_a = 'A' if rf_prediction == 'a' else 'inactive'
        room_b = 'B' if rf_prediction == 'b' else 'inactive'
        room_c = 'C' if rf_prediction == 'c' else 'inactive'
        data = {
            'rooms': {'A': room_a, 'B': room_b, 'C': room_c},
            'classifiers': [{'name': 'Random Forest', 'accuracy': rf_accuracy}, {'name': 'KNN', 'accuracy': knn_accuracy}]
        }
        print(data)
        if train == False:
            result_A = []
            result_B = []
            result_C = []
        return data
    
##for testing we need to get the last ele
    
def calculate_majority_vote(k, classifier_index=0):
    global history
    vote_counts = {'a': 0, 'b': 0, 'c': 0}
    
    print("history: " + str(history))
    for predictions in history:
        prediction = predictions[classifier_index]
        print(f"Raw prediction: {prediction}, Type: {type(prediction)}")  
        prediction_key = prediction.item() if isinstance(prediction, np.ndarray) else prediction
        print(f"Processed prediction key: {prediction_key}, Type: {type(prediction_key)}")  
        if prediction_key in vote_counts:
            vote_counts[prediction_key] += 1
        else:
            print(f"Key not found in vote_counts: {prediction_key}") 

    print("Final vote counts:", vote_counts)
    majority_room = max(vote_counts, key=lambda x: (vote_counts[x], -('abc'.index(x))), default=None)
    
    if majority_room is None or vote_counts[majority_room] == 0:
        return json.dumps({})

    data = {
        'majority_room': majority_room.capitalize(),
        'strength': vote_counts[majority_room]
    }
    json_string = json.dumps(data)
    return json_string
