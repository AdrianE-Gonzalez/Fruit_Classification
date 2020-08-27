import tensorflow as tf
import keras_preprocessing
import keras
import numpy as np
from keras.optimizers import Adamax, Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.models import load_model

#Currently Created A Template Of How The NN.py Will Be Structured With The Use Of Examples From Keras Documentation

#Ideas For Updating NN.py:
#       -The Ability To Choose The Parameters For The Model (In The Long Run)
#       -Create A Suitable Algorithm That Is Able To Process The Data With Greater Than 50% Accuracy


#Might Use Augmentation, Need To Read Up On IT More Thoroughly
def augmented_train_data(train_data):
    augmented_train_data= train_data.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    )
    return augmented_train_data

#Creates Basic Model Based On Keras Example
def creating_model(train_data):
    print("Creating Model")
    num_classes = len(train_data.class_names)

    model = tf.keras.Sequential()
    
    model.add(keras.layers.experimental.preprocessing.Rescaling(1./255)),
    model.add(Conv2D(32, (3, 3), input_shape=(3, 160, 160)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    

    print("Compiling Model")
    # compile the keras model
    model.compile(optimizer = Adamax(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),loss='sparse_categorical_crossentropy', metrics =['accuracy'])
    print("Done Creating Model")
    return model

#Saves Model For Later Use
def save_model(model):
    model.save("model_3.h5")
    print("Model Saved")

#Loads Model That Was Previously Saved
def load_model_():
    model = load_model('model_2.h5')
    print("Model Loaded")

    return model

#Uses Create_Model And Trains The Dataset In Use; Returns The Model.Fit For Plotting Accuracies
def run_model(train_data,val_data):
    print("Running Model")
    model=creating_model(train_data)
    
    bs = 25
    hist = model.fit(train_data, epochs=bs, batch_size=bs, validation_data=(val_data))

    save_model(model)

    print("Done Running Model")
    return hist

#This Will Make Predictions Based On Image Given
def nn_prediction(model,pred,class_names):
    predictions = model.predict(pred)
    # Generate arg maxes for predictions
    classes = np.argmax(predictions, axis = -1)
    print("The Image Prediction Is: "+class_names[int(classes)])
    print("Prediction Complete")
print("neural networks code")