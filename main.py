# random forest model 

import pandas as pd 
import numpy as np 

df = pd.read_csv("heart_disease.csv")
print(df.head()) 
print(df.info())

X = df.drop(["target"], axis=1) # Hide target and metrics 
Y = df["target"] # target 

#Split into training and testing 
from sklearn.model_selection import train_test_split 

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y, test_size = 0.2, random_state = 42
)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    max_depth=8, 
    min_samples_leaf=5, 
   random_state = 42
)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, Y_pred))

from sklearn.metrics import classification_report 

print(classification_report(Y_test, Y_pred))

# Check if overfitting or underfitting 

train_accuracy  = model.score(X_train, Y_train)
test_accuracy = model.score(X_test, Y_test)


print("Train:", train_accuracy)   
print("Test:", test_accuracy)
# Results showed slight overfitting, needed to account for this 


# Make predictions: 
results = pd.DataFrame({
    "Actual": Y_test.values,
    "Predicted": Y_pred
})
results.to_csv("predictions.csv")