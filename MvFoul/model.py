from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, Flatten

def create_model(input_shape=(16, 224, 224, 3)):
    model = Sequential([
        Input(shape=input_shape),
        Conv3D(32, (3, 3, 3), activation='relu'),
        MaxPooling3D((1, 2, 2)),
        Conv3D(64, (3, 3, 3), activation='relu'),
        MaxPooling3D((1, 2, 2)),
        Conv3D(64, (3, 3, 3), activation='relu'),
        MaxPooling3D((1, 2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model