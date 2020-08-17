import tensorflow as tf
import keras_preprocessing
import keras
from keras.optimizers import Adamax
from keras.models import Sequential

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
    model.add(keras.layers.Conv2D(32, 3, activation='relu')),
    model.add(keras.layers.MaxPooling2D()),
    model.add(keras.layers.Conv2D(32, 3, activation='relu')),
    model.add(keras.layers.MaxPooling2D()),
    model.add(keras.layers.Conv2D(32, 3, activation='relu')),
    model.add(keras.layers.MaxPooling2D()),
    model.add(keras.layers.Flatten()),
    model.add(keras.layers.Dense(128, activation='relu')),
    model.add(keras.layers.Dense(num_classes))
    

    print("Compiling Model")
    # compile the keras model
    model.compile(optimizer = Adamax(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),loss='sparse_categorical_crossentropy', metrics =['accuracy'])
    print("Done Creating Model")
    return model

#Uses Create_Model And Trains The Dataset In Use; Returns The Model.Fit For Plotting Accuracies
def run_model(train_data,val_data):
    print("Running Model")
    model=creating_model(train_data)
    
    bs = 25
    hist = model.fit(train_data, epochs=25, batch_size=bs, validation_data=(val_data))

    print("Done Running Model")
    return hist

print("neural networks code")

