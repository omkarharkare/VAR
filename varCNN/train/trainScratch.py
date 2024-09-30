## Train your model form scratch 
## Change the model and it's properties byb editing model.py and dataLoad.py

import tensorflow as tf
import numpy as np 
from model import create_model
from dataLoad import load_data

path = "D:/App Dev/flashcards/git/Var/data/var600/VAR"

model = create_model() #By default the input shape is (256,256,3)

train,val=load_data(path) #laoding the train and validation data

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)

# Compile your model 
#model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])
model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])

#Define the EarlyStopping Callback to refrain from overfitting patience range standard=(10,20)
es=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=0,restore_best_weights=True)

#start training
#history=model.fit(train, validation_data=val,epochs=110,callbacks=[es])
history = model.fit(train, validation_data=val, epochs=50, callbacks=[es, lr_reducer])
#training steps and validation steps are calculated automatically
#Train for less epochs if not on GPU

