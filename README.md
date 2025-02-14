# Multivariate Time Series Forecasting using GRU (Gated Recurrent Units)

This project demonstrates how to forecast multivariate time series data using GRU (Gated Recurrent Units), a type of Recurrent Neural Network (RNN). The model predicts wind speed based on various meteorological factors such as temperature, precipitation, and wind indicators. The dataset used comes from the [Wind Speed Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/wind-speed-prediction-dataset).

---

## Dataset Overview

### Source:
The dataset is provided by [Fedesoriano](https://www.kaggle.com/datasets/fedesoriano/wind-speed-prediction-dataset) and can be downloaded from Kaggle.

### Attributes:
- **DATE**: Date of the observation (YYYY-MM-DD)
- **WIND**: Average wind speed in knots
- **IND**: First indicator value
- **RAIN**: Precipitation amount in mm
- **IND.1**: Second indicator value
- **T.MAX**: Maximum temperature in Celsius
- **IND.2**: Third indicator value
- **T.MIN**: Minimum temperature in Celsius
- **T.MIN.G**: Grass minimum temperature at 09 UTC in Celsius

### Dataset Preparation:

1. Load the dataset and inspect it.
2. Drop any missing values.
3. Convert the date column to datetime format and set it as the index.

```python
dataset = pd.read_csv(r"path_to_your_dataset.csv")
dataset['DATE'] = pd.to_datetime(dataset['DATE'])
dataset.set_index('DATE', inplace=True)
```

---

## Preprocessing Steps

### Step 1: Visualize the Dataset

Plot the dataset's features to observe trends and relationships.

```python
plt.figure(figsize=(10, 10))
for group in range(len(dataset.columns)):
    plt.subplot(len(dataset.columns), 1, group + 1)
    plt.plot(dataset.values[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
plt.show()
```

### Step 2: Series to Supervised Learning Conversion

Convert the time series data into a supervised learning problem, where past observations are used to predict future values.

```python
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # Convert series to supervised learning format
    # ...
```

### Step 3: Normalize Features

Normalize the features using MinMaxScaler to scale values between 0 and 1.

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset.values)
```

### Step 4: Encoding Categorical Data

If there are categorical features (e.g., temperature indicators), encode them using `LabelEncoder`.

```python
encoder = LabelEncoder()
dataset['T.MAX'] = encoder.fit_transform(dataset['T.MAX'])
```

### Step 5: Train-Test Split

Split the dataset into training and testing sets. We use 80% of the data for training and 20% for testing.

```python
n_train_days = 365 * 14
train = scaled[:n_train_days, :]
test = scaled[n_train_days:, :]
```

---

## Model Architecture

### GRU-based Model

The model is built using a GRU architecture, where the model is trained to predict the wind speed based on past observations of various meteorological variables.

```python
def build_gru_model(n_layers, n_units, input_shape):
    model = Sequential()
    for _ in range(n_layers):
        model.add(GRU(n_units, input_shape=input_shape, return_sequences=True))
    model.add(Dense(1))  # Output layer
    model.summary()
    return model
```

Two models are tested:
- **Single Layer GRU** with 50 units
- **Two Layers GRU** with 64 units

---

## Training the Model

### Step 1: Compile the Model

The models are compiled using the Adam optimizer and Mean Absolute Error (MAE) loss.

```python
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
```

### Step 2: Train the Models

The models are trained for 50 epochs with a batch size of 72, using 10% of the data as a validation set.

```python
history_single_layer = single_layer_gru_model.fit(Xtrain, Ytrain, epochs=50, batch_size=72, validation_split=0.1, verbose=1)
history_two_layers = two_layers_gru_model.fit(Xtrain, Ytrain, epochs=50, batch_size=72, validation_split=0.1, verbose=1)
```

### Step 3: Plot Training and Validation Loss

Plot the training and validation loss to evaluate the model performance during training.

```python
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and Validation loss for single layer')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

## Results

### Step 1: Evaluate the Models on Test Data

After training, evaluate the models on the test data.

```python
testResult_single_layer = single_layer_gru_model.evaluate(Xtest, Ytest)
testResult_two_layers = two_layers_gru_model.evaluate(Xtest, Ytest)
```

### Step 2: Print the Results

Print the evaluation results for both models.

```python
print(testResult_single_layer)
print(testResult_two_layers)
```

---

## Conclusion

In this project, we built and trained two GRU-based models for forecasting wind speed using a variety of meteorological features. By visualizing the training loss and evaluating model performance on test data, we can gauge the effectiveness of our models. Future improvements may include tuning hyperparameters, adding more features, or experimenting with other types of RNNs (e.g., LSTM).

---

## Dependencies

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn

Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

---

This project is a basic implementation of GRU for multivariate time series forecasting. You can adapt this code to your own time series prediction tasks by replacing the dataset and modifying the model architecture or hyperparameters as needed.

## 🌍 Explore More Projects  
For more exciting machine learning and AI projects, visit **[The iVision](https://theivision.wordpress.com/)** and stay updated with the latest innovations and research! 🚀  

---
