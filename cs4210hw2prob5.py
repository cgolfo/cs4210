from sklearn.naive_bayes import GaussianNB
import csv

# Read the training data from a CSV file
data_train = []
labels_train = []

with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read the header
    for row in reader:
        data_train.append([row[1], row[2], row[3], row[4]])  # Features: Outlook, Temperature, Humidity, Wind
        labels_train.append(row[5])  # Class: PlayTennis

# Transform the original training features to numbers
# You can use a mapping for feature values (e.g., Sunny = 1, Overcast = 2, Rain = 3)
feature_mapping = {'Sunny': 1, 'Overcast': 2, 'Rain': 3, 'Hot': 1, 'Mild': 2, 'Cool': 3, 'High': 1, 'Normal': 2, 'Weak': 1, 'Strong': 2}

X_train = [[feature_mapping[feature] for feature in instance] for instance in data_train]
Y_train = [1 if label == 'Yes' else 2 for label in labels_train]

# Fitting the Naïve Bayes classifier to the training data
clf = GaussianNB()
clf.fit(X_train, Y_train)

# Read the test data from a CSV file
data_test = []

with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read the header
    for row in reader:
        data_test.append([row[1], row[2], row[3], row[4]])  # Features: Outlook, Temperature, Humidity, Wind

# Print the header for the solution
print("Day Outlook Temperature Humidity Wind PlayTennis Confidence")

# Use the test samples to make probabilistic predictions
for i, test_instance in enumerate(data_test):
    X_test = [feature_mapping[feature] for feature in test_instance]
    probabilities = clf.predict_proba([X_test])[0]
    confidence = max(probabilities)

    # Determine the predicted class based on the class with higher probability
    predicted_class = 'Yes' if probabilities[0] > probabilities[1] else 'No'

    # Output the classification if confidence is >= 0.75
    if confidence >= 0.75:
        print(f"D{i+1} {' '.join(test_instance)} {predicted_class} {confidence:.2f}")
