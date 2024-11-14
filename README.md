# Module 13 Challenge

## Spam Predictions

This notebook uses machine learning to detect spam mail.

### Requirements

#### Split the Data into Training and Testing Sets

* There is a prediction about which model you expect to do better.

``` python
"""
I believe that the Random Forests Classifier should be better. Random Forest is less prone to overfitting hence increasings its accuracy and stability. It can handle non linear relationships between all the input variables as well. Also, it  handles imbalanced data sets better, which this data set has almost 1000 more non spam data points than spam datapoints.
"""
```

* The labels set (y) is created from the “spam” column.

``` python
y = data["spam"]
```

* The features DataFrame (X) is created from the remaining columns.

``` python
X = data.drop(columns="spam")
```

* The value_counts function is used to check the balance of the labels variable (y).

``` python
y.value_counts()
```

* The data is correctly split into training and testing datasets by using train_test_split.

``` python
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

#### Scale the Features

* An instance of StandardScaler is created.

``` python
scaler = StandardScaler()
```

* The Standard Scaler instance is fit with the training data.

``` python
scaler.fit(X_train)
```

* The training features DataFrame is scaled using the transform function.

``` python
X_train_scaled = scaler.transform(X_train)
```

* The testing features DataFrame is scaled using the transform function.

``` python
X_test_scaled = scaler.transform(X_test)
```

#### Create a Logistic Regression Model 

* A logistic regression model is created with a random_state of 1.

``` python
lrm = LogisticRegression(random_state=1)
```

* The logistic regression model is fitted to the scaled training data (X_train_scaled and y_train).

``` python
lrm.fit(X_train_scaled,y_train)
```

* Predictions are made for the testing data labels by using the testing feature data (X_test_scaled) and the fitted model, and saved to a variable.

``` python
lrm_testing_predictions = lrm.predict(X_test_scaled)
```

* The model’s performance is evaluated by calculating the accuracy score of the model with the accuracy_score function.

``` python
accuracy_score(y_test, lrm_testing_predictions)
```

#### Create a Random Forest Model

* A random forest model is created with a random_state of 1. 

``` python
rfc = RandomForestClassifier(random_state=1)
```

* The random forest model is fitted to the scaled training data (X_train_scaled and y_train).

``` python
rfc.fit(X_train_scaled,y_train)
```

* Predictions are made for the testing data labels by using the testing feature data (X_test_scaled) and the fitted model, and saved to a variable.

``` python
rfc_testing_predictions = rfc.predict(X_test_scaled)
```

* The model’s performance is evaluated by calculating the accuracy score of the model with the accuracy_score function.

``` python
accuracy_score(y_test, rfc_testing_predictions)
```

#### Evaluate the Models

* Which model performed better?

``` python
"""
The logistic regression actually did better than the random forest.
"""
```

* How does that compare to your prediction?

``` python
"""
It was not per my prediction.
This is because even though random forest has a higher score for the test data, the difference between the train data and test data is nearly 5%. Whereas, for the logistic regression model, the difference was only about 1%.
"""
```