
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Set non-interactive backend for matplotlib
plt.switch_backend('agg')

# Create encoder
encoder = LabelEncoder()

# Load data
iris = pd.read_csv('Iris.csv')

# Encode target variable
iris['Species'] = encoder.fit_transform(iris['Species'])

# Prepare inputs and target
inputs = iris.drop(['Id', 'Species'], axis=1)
target = iris['Species']

# Train decision tree
dec_tree = tree.DecisionTreeClassifier(random_state=0)
classifier = dec_tree.fit(inputs, target)

# Save the model
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Plot and save the tree visualization
plt.figure(figsize=(20, 15))
tree.plot_tree(classifier, filled=True, feature_names=inputs.columns)
plt.savefig('decision_tree.png')

# Print accuracy score
print(f"Model training completed. Accuracy on training data: {dec_tree.score(inputs, target):.4f}")

#### TASK:
We have to build a decision tree that can effectively make suitable classifications, accurately, based on graphical visualizations.

#### APPROACH:
We could define the task assigned into two parts. Part one could comprise of a Decision Tree model that trains itself based on the provided data and make predictions while the second part could contain a more diverse function that enables us to plot the tree for any data. In this, we could define a function that takes the input of the input and target variables and directly plots the graph and makes classification based predictions. We can load iris as an in-built dataset but since we have access to the csv file in the local system directory, we will import the dataset from the local directory itself.
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder as le
# %matplotlib inline

iris = pd.read_csv('/content/Iris.csv')
iris

iris['Species'] = le.fit_transform(iris['Species'], iris['Species'])
iris

inputs = iris.drop(iris[['Id', 'Species']], axis =1)
inputs

target = iris[['Species']]
target

dec_tree = tree.DecisionTreeClassifier(random_state = 0)

classifier = dec_tree.fit(inputs, target)

plt.figure(figsize=(20,15))
tree.plot_tree(classifier, filled = True)

