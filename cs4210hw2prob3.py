import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import csv

data = []
labels = []

with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            try:
                label = int(row[2])
                data.append([float(row[0]), float(row[1])])
                labels.append(label)
            except ValueError:
                print(f"Skipped row {i+1} due to invalid label: {row[2]}")

data = np.array(data)
labels = np.array(labels)

# Create a meshgrid of points covering the range of the data
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Create a 1-NN classifier and fit it to the data
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(data, labels)

# Predict the class for each point in the meshgrid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)

# Plot the data points
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.RdYlBu, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary for 1-NN')
plt.show()
