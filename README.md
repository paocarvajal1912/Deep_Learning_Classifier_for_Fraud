# Predicting Fraudulent Transactions Using Neural Networks

Here you will find a classification deep learning model to predict fraudulent transactions.

## Steps:

### I. Load and Preprocess Data

 We read the data from the CSV file provided in the "Resources" folder, create the target set, and define the features.
 Then, we used the `train_test_split` function from scikit-learn to create the training and testing sets.
 Finally for the first step, we scale the features data using the `StandardScaler` from `scikit-learn.`

### II. Create and Evaluate a Deep Neural Network Model

We compiled the neural network model using the `binary_crossentropy` loss function, the `adam` optimizer, and `accuracy` as an additional metric. We fit the model with the training data, using 100 epochs.

Finally we evaluate the model using testing data and the `evaluate` method.

-----

# Technologies

For the neural network model we use `TensorFlow` and `Keras`. Particularly `Sequential` for generating a linear stack of layers, and`Dense` for the internal definition of the layers itself. Also, we use the `evaluate` method of Keras to measure the performance of the model.

For the preprocessing of the data we used `StandardScaler`from the `preprocessing` module of `Sklearn`. This means that the data used to train the model was standarized to mean equal zero and standard deviation equal one before using it.

For the separation of the train data and test data groups we used `train_test_split` function from the `model_selection` module of `Sklearn`. We used 75% of the data to train the model, and 25% to test it after.



## References:

[Keras Sequential model](https://keras.io/api/models/sequential/)

[Keras Dense module](https://keras.io/api/layers/core_layers/dense/)

[Keras evaluate](https://keras.io/api/models/model_training_apis/)
