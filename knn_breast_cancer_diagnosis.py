import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

# Data Preprocessing
# Remove the 'Unnamed: 32' and 'id' columns from the DataFrame
df.drop(columns=['Unnamed: 32', 'id'], inplace=True, errors='ignore')

# Splitting the data into training and test sets to train and evaluate the model
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_neighbors': [3, 5, 7],  # Trying different numbers of neighbors
    'weights': ['uniform', 'distance'],  # Considering uniform or distance-based weighting
    'p': [1, 2]  # Trying out different distance metrics
}

# Creating the K-Nearest Neighbors classifier and GridSearchCV instance
knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')

# Fitting the training data to perform the grid search and hyperparameter tuning
grid_search.fit(X_train, y_train)

# Obtaining the best parameters and best score achieved during the search
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the best parameters and best score
print("Best Parameters:", best_params)
print("Best Score:", best_score)
