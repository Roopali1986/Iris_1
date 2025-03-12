
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





