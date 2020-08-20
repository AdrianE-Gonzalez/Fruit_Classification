import load_data
import matplotlib.pyplot as plt
import nn
#Main Python File Where It Uses All The Files Created To Run Code
#Planned Algorithms NN, Decision Tree,...

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

print("Main Code")
train_data,val_data,test_data= load_data.load_datasets()

hist = nn.run_model(train_data,val_data)

plot_data(hist)
#print(train_data.class_names)
#print(val_data.class_names)

print("Done")