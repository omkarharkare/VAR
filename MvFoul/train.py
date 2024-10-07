import tensorflow as tf
from preprocess import prepare_data
from data_loader import VideoDataGenerator
from model import create_model

# Set your root directory
root_dir = 'SoccerNet/mvFouls'

# Prepare data
train_data, train_labels = prepare_data(root_dir, 'train')
val_data, val_labels = prepare_data(root_dir, 'valid')
test_data, test_labels = prepare_data(root_dir, 'test')

# Create data generators
train_gen = VideoDataGenerator(train_data, train_labels, batch_size=32)
val_gen = VideoDataGenerator(val_data, val_labels, batch_size=32)
test_gen = VideoDataGenerator(test_data, test_labels, batch_size=32)

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks,
    steps_per_epoch=len(train_gen),
    validation_steps=len(val_gen)
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the final model
model.save('final_model.h5')