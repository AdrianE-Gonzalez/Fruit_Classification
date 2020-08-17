import tensorflow as tf
import os
from dotenv import load_dotenv
#Currently Only Uses Fruit_360 Dataset
#For FILE_PATH, You'll Have To Create An External .env File That Stores FILE_PATH
#   Ex: Open Notepad and Save It As A .env File
#           Inside The .env File Write:
#                 FILE_PATH= 'YOUR FILE_PATH WHERE THE DATSET IS STORED'

#Ideas For Updating Load_Data.py
#       -Optimize For Any Type Of Dataset
#       -Be Able TO Load Dataset Without The A Set FILEPATH

def load_datasets():
    print("Loading Datasets Into Training, Validation and Testing")
    #Loads Set File Directory Of The Dataset
    load_dotenv()
    FILE_PATH=os.getenv('FILE_PATH')

    #Testing And Training Directories
    test_dir= FILE_PATH+'Test'
    train_dir= FILE_PATH+'Training'

    #Create Image Data Generator With Image Augmentation
    image_size = 160
    batch_size = 32

    #Loads Testing, Validation And Training Datasets Into A 'tf.data.Dataset' Object
    #Labels: Label Parameter Is Set To Inferred In Order To Indicate Subdirectories
    #      Ex:  main_directory/
    #           ...class_a/
    #           ......a_image_1.jpg
    #           ......a_image_2.jpg
    #           ...class_b/
    #           ......b_image_1.jpg
    #Label_Mode: Label_Mode Parameter Is Set To Int To Indicate The Use For Sparse_Categorial_Crossentropy
    #                   Can Also Be Set To 'Categorial' For Categorial_Crossentropy And 'Binary' For Binary_Crossentropy
    #Validation_Split: Validation_Split Parameter Reserves Data For Validation, A Float Number between 0-1
    #Subset: Subset Parameter Identifies Whether The Data In Use Is 'Training' or 'Validation', Can Only Be Set If A Validation_Split Is Defined
    #Seed: Seed Parameter Can Be Set To Any Number As Long As It's The Same For Both Training And Validation
    #      It's Used For Splitting The Data And Eliminate Overlapping Of Data Between Training And Validating
    test_dataset=tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, 
        labels='inferred'
    )

    train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, 
        labels='inferred',
        label_mode='int',
        validation_split=0.3,
        subset='training',
        seed=1337,
        image_size=(image_size,image_size),
        batch_size=batch_size
    )
    val_dataset=tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, 
        labels='inferred',
        label_mode='int',
        validation_split=0.3,
        subset='validation',
        seed=1337,
        image_size=(image_size,image_size),
        batch_size=batch_size
    )
    print("Loading Complete")
    return train_dataset, val_dataset, test_dataset