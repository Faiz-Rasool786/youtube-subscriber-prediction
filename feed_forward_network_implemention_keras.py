import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# Load data
csv_path = r"D:\Faizi\Neural Network\Keras Sequential API - MNIST Dataset\content\List of most-subscribed YouTube channels.csv"
data = pd.read_csv(csv_path)

# Strip column names
data.columns = data.columns.str.strip()

# Clean target column
data['Subscribers (millions)'] = pd.to_numeric(data['Subscribers (millions)'], errors='coerce')
data = data.dropna(subset=['Subscribers (millions)'])

# Select relevant columns
df = data[['Subscribers (millions)', 'Category', 'Country']].copy()

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Category', 'Country'], drop_first=True)

# Separate features and target
X = df.drop('Subscribers (millions)', axis=1)
y = df['Subscribers (millions)']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature values
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build the model
model = Sequential()
model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Save model weights during training
filepath = r'D:\Faizi\Neural Network\Keras Sequential API - MNIST Dataset\model_output\weights-{epoch:02d}-{loss:.4f}.h5'
checkpoint = ModelCheckpoint(filepath=filepath, save_weights_only=True, monitor='val_loss', mode='min',
                             save_best_only=False, save_freq='epoch')

# Custom callback for R² score
class MyCustomCallback(Callback):
    def __init__(self, features, target):
        self.test_data = (features, target)
        self.r2_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.test_data[0])
        r2 = r2_score(self.test_data[1], y_pred)
        print(f'\nEpoch {epoch+1}: R² Score on validation set = {r2:.4f}\n')
        self.r2_scores.append((epoch + 1, r2))

# Instantiate custom callback
mycallback = MyCustomCallback(x_test, y_test)

# Print model summary
model.summary()

# Train the model
history = model.fit(x_train, y_train,
                    validation_split=0.2,
                    epochs=200,
                    batch_size=4,
                    verbose=1,
                    callbacks=[early_stop, checkpoint, mycallback])

# Evaluate the model
loss, mae = model.evaluate(x_test, y_test)
print(f"\nFinal Evaluation on Test Set:\nTest Loss (MSE): {loss:.4f}, Test MAE: {mae:.4f}")
