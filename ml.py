import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from collections import deque


def load_and_combine_csv(filenames, names):
    dataframes = []

    # Load and combine CSV files with column names
    for filename, name in zip(filenames, names):
        df = pd.read_csv(
            filename, header=None, names=["feature1", "feature2", "feature3"]
        )
        df["source"] = name
        dataframes.append(df)

    # Combine all dataframes into one
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def predict(model, X_test):
    return model.predict(X_test)


def evaluate_predictions(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100


def main():
    train_filenames = ["a.csv", "b.csv", "c.csv"]
    # test_filenames = ["a_test.csv", "b_test.csv", "c_test.csv"]
    names = ["a", "b", "c"]

    # Load and combine data
    train_data = load_and_combine_csv(train_filenames, names)
    # test_data = load_and_combine_csv(test_filenames, names)

    # Split data into features and target
    X_train = train_data.drop("source", axis=1)
    y_train = train_data["source"]
    # X_test = test_data.drop("source", axis=1)
    # y_test = test_data["source"]

    # Training
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train, n_neighbors=3)

    # Predictions
    # rf_predictions = predict(rf_model, X_test)
    # knn_predictions = predict(knn_model, X_test)

    # Evaluate RF model
    # rf_accuracy = evaluate_predictions(y_test, rf_predictions)
    # print("Random Forest Accuracy:", rf_accuracy)
    # print("Random Forest Predictions:", rf_predictions)

    # Evaluate KNN model
    # knn_accuracy = evaluate_predictions(y_test, knn_predictions)
    # print("KNN Accuracy:", knn_accuracy)
    # print("KNN Predictions:", knn_predictions)

    # Save the models
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(knn_model, 'knn_model.pkl')

if __name__ == "__main__":
    main()
