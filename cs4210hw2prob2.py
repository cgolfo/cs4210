from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Function to transform categorical data to numbers
def transform_category_to_number(category, category_mapping):
    if category not in category_mapping:
        category_mapping[category] = len(category_mapping) + 1
    return category_mapping[category]

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # Skipping the header
                dbTraining.append(row)

    # Initialize category mappings
    category_mapping_X = {}
    category_mapping_Y = {}

    # Transform the original categorical training features to numbers and add to the 4D array X
    for row in dbTraining:
        X_row = [transform_category_to_number(category, category_mapping_X) for category in row[:-1]]
        X.append(X_row)

    # Transform the original categorical training classes to numbers and add to the vector Y
    Y = [transform_category_to_number(row[-1], category_mapping_Y) for row in dbTraining]

    # Initialize variables to calculate the average accuracy
    total_accuracy = 0

    for _ in range(10):
        # Fitting the decision tree to the data, setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        dbTest = []

        # Read the test data and add it to dbTest
        with open('contact_lens_test.csv', 'r') as test_csvfile:
            test_reader = csv.reader(test_csvfile)
            for i, test_row in enumerate(test_reader):
                if i > 0:
                    dbTest.append(test_row)

        correct_predictions = 0

        for test_data in dbTest:
            X_test = [transform_category_to_number(category, category_mapping_X) for category in test_data[:-1]]
            class_predicted = clf.predict([X_test])[0]

            # Transform the true label to a number
            true_label = transform_category_to_number(test_data[-1], category_mapping_Y)

            # Compare the prediction with the true label
            if class_predicted == true_label:
                correct_predictions += 1

        # Calculate accuracy for this iteration
        accuracy = correct_predictions / len(dbTest)

        # Add accuracy to total_accuracy
        total_accuracy += accuracy

    # Calculate the average accuracy for the model over 10 runs
    average_accuracy = total_accuracy / 10

    # Print the average accuracy for this model
    print(f'Final accuracy when training on {ds}: {average_accuracy:.2f}')

    
