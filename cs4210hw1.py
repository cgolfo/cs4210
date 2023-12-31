#-------------------------------------------------------------------------
# AUTHOR: Carlo Golfo
# FILENAME: CS4210hw1.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 Hours
#-----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays
#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []
#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)
            print(row)
#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
#--> add your Python code here
# X =
age_mapping = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_mapping = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_mapping = {'Yes': 1, 'No': 2}
tear_mapping = {'Reduced': 1, 'Normal': 2}

X = []
for row in db:
    age = age_mapping[row[0]]
    spectacle = spectacle_mapping[row[1]]
    astigmatism = astigmatism_mapping[row[2]]
    tear = tear_mapping[row[3]]

    # Append the transformed data as a list to X
    X.append([age, spectacle, astigmatism, tear])
    print(X) 

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
# Y =
class_mapping = {'Yes': 1, 'No': 2}
Y = []
for row in db: 
    Y.append(class_mapping[row[-1]]) 
#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)
#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'],
class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
