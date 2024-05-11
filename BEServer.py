from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import json
from time import sleep
import random
import socket

# Assume preprocessing.py contains these functions
from preprocessing import collect_data, collect_test_data, calculate_majority_vote

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allows all origins for all routes

random.seed(42)

train = False
stream_data = {}

def update_stream_data(new_data):
    global stream_data
    stream_data = new_data

@app.route('/data', methods=['POST', 'GET'])
def data():
    try:
        if request.method == 'POST':
            data = request.get_data().decode()
            print("Received data:", data)
            if train:
                collect_data(data) 
            else: 
                processed_data = collect_test_data(data)
                update_stream_data(processed_data)
            return jsonify({"message": "Data received"}), 200
        elif request.method == 'GET':
            return Response(stream_generator(), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def stream_generator():
    global stream_data
    old_data = None
    while True:
        if old_data != stream_data:
            yield f'data: {json.dumps(stream_data)}\n\n'
            old_data = stream_data
        sleep(1)

@app.route('/set_classifier', methods=['POST'])
def set_classifier():
    try:
        global current_classifier_index
        data = request.json
        classifier_map = {'Random Forest': 0, 'KNN': 1}
        current_classifier_index = classifier_map.get(data.get('classifier', 'Random Forest'), 0)
        return jsonify({"message": "Classifier updated"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream_majority_vote', methods=['GET'])
def stream_majority_vote():
    try:
        k = int(request.args.get('k', 20))
        classifier_name = request.args.get('classifier', 'Random Forest')
        classifier_map = {'Random Forest': 0, 'KNN': 1}
        classifier_index = classifier_map.get(classifier_name, 0)

        return Response(majority_vote_generator(k, classifier_index), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def majority_vote_generator(k, classifier_index):
    while True:
        majority_vote_data = calculate_majority_vote(k, classifier_index)
        yield f'data: {majority_vote_data}\n\n'
        sleep(1)

if __name__ == '__main__':
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    app.run(host='0.0.0.0', port=5001, debug=True)