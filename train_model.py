# ----------------- IMPORT LIBRARIES -----------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ----------------- STEP 1: LOAD DATA -----------------
data = pd.read_csv('landmarks_dataset.csv')

# X = all 63 landmarks, y = label
X = data.drop('label', axis=1).values
y = data['label'].values

# Encode labels (A→0, B→1 ... )
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ----------------- STEP 2: SPLIT DATA -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# ----------------- STEP 3: BUILD MODEL -----------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')
])

# ----------------- STEP 4: COMPILE MODEL -----------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ----------------- STEP 5: TRAIN MODEL -----------------
model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=50,
          batch_size=32)

# ----------------- STEP 6: SAVE MODEL -----------------
model.save('ae_hand_model.h5')

print("Model Trained & Saved Successfully!")
