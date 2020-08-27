import load_data
import matplotlib.pyplot as plt
import nn
from keras.preprocessing.image import load_img
#Main Python File Where It Uses All The Files Created To Run Code
#Planned Algorithms CNN, Decision Tree,...

#Plots Accuracies For Model Accuracy And Model Loss
def plot_data(hist):
    #summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')

    plt.show()

    #summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

def load_single_image():
    #class_names= load_data.load_class_names()
    #samp_path='ADD SINGLE IMAGE PATH HERE'
    #sample= load_data.load_single_img(samp_path)
    return "Finish Loading Image"

def run_prediction():
    #pred =nn.load_model_()
    #nn.nn_prediction(pred,sample,class_names)
    return "Prediciton Complete"

def run_test():
    #Uncomment Following Lines For Training Purposes
    #train_data,val_data,test_data= load_data.load_datasets()
    #hist = nn.run_model(train_data,val_data)

    #plot_data(hist)
    return "Training Complete"

print("Main Code")
