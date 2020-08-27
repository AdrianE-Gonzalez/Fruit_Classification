import tensorflow as tf
import keras
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

#Loads A Single Image For Testing
def load_single_img(path):
    img = keras.preprocessing.image.load_img(
        path, target_size=(160, 160)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    return img_array

#Hard Coded The List Of Fruits Relative To The Order It Appears
def load_class_names():
    class_names= {
        0: 'Apple Braeburn',
        1: 'Apple Crimson Snow',
        2: 'Apple Golden 1',
        3: 'Apple Golden 2',
        4: 'Apple Golden 3',
        5: 'Apple Granny Smith',
        6: 'Apple Pink Lady',
        7: 'Apple Red 1',
        8: 'Apple Red 2',
        9: 'Apple Red 3',
        10: 'Apple Red Delicious',
        11: 'Apple Red Yellow 1',
        12: 'Apple Red Yellow 2',
        13: 'Apricot',
        14: 'Avocado',
        15: 'Avocado ripe',
        16: 'Banana',
        17: 'Banana Lady Finger',
        18: 'Banana Red',
        19: 'Beetroot',
        20: 'Blueberry',
        21: 'Cactus fruit',
        22: 'Cantaloupe 1',
        23: 'Cantaloupe 2',
        24: 'Carambula',
        25: 'Cauliflower',
        26: 'Cherry 1',
        27: 'Cherry 2',
        28: 'Cherry Rainier',
        29: 'Cherry Wax Black',
        30: 'Cherry Wax Red',
        31: 'Cherry Wax Yellow',
        32: 'Chestnut',
        33: 'Clementine',
        34: 'Cocos',
        35: 'Corn',
        36: 'Corn Husk',
        37: 'Cucumber Ripe',
        38: 'Cucumber Ripe 2',
        39: 'Dates',
        40: 'Eggplant',
        41: 'Fig',
        42: 'Ginger Root',
        43: 'Granadilla',
        44: 'Grape Blue',
        45: 'Grape Pink',
        46: 'Grape White',
        47: 'Grape White 2',
        48: 'Grape White 3',
        49: 'Grape White 4',
        50: 'Grapefruit Pink',
        51: 'Grapefruit White',
        52: 'Guava',
        53: 'Hazelnut',
        54: 'Huckleberry',
        55: 'Kaki',
        56: 'Kiwi',
        57: 'Kohlrabi',
        58: 'Kumquats',
        59: 'Lemon',
        60: 'Lemon Meyer',
        61: 'Limes',
        62: 'Lychee',
        63: 'Mandarine',
        64: 'Mango',
        65: 'Mango Red',
        66: 'Mangostan',
        67: 'Maracuja',
        68: 'Melon Piel de Sapo',
        69: 'Mulberry',
        70: 'Nectarine',
        71: 'Nectarine Flat',
        72: 'Nut Forest',
        73: 'Nut Pecan',
        74: 'Onion Red',
        75: 'Onion Red Peeled',
        76: 'Onion White',
        77: 'Orange',
        78: 'Papaya',
        79: 'Passion Fruit',
        80: 'Peach',
        81: 'Peach 2',
        82: 'Peach Flat',
        83: 'Pear',
        84: 'Pear 2',
        85: 'Pear Abate',
        86: 'Pear Forelle',
        87: 'Pear Kaiser',
        88: 'Pear Monster',
        89: 'Pear Red',
        90: 'Pear Stone',
        91: 'Pear Williams',
        92: 'Pepino',
        93: 'Pepper Green',
        94: 'Pepper Orange',
        95: 'Pepper Red',
        96: 'Pepper Yellow',
        97: 'Physalis',
        98: 'Physalis with Husk',
        99: 'Pineapple',
        100: 'Pineapple Mini',
        101: 'Pitahaya Red',
        102: 'Plum',
        103: 'Plum 2',
        104: 'Plum 3',
        105: 'Pomegranate',
        106: 'Pomelo Sweetie',
        107: 'Potato Red',
        108: 'Potato Red Washed',
        109: 'Potato Sweet',
        110: 'Potato White',
        111: 'Quince',
        112: 'Rambutan',
        113: 'Raspberry',
        114: 'Redcurrant',
        115: 'Salak',
        116: 'Strawberry',
        117: 'Strawberry Wedge',
        118: 'Tamarillo',
        119: 'Tangelo',
        120: 'Tomato 1',
        121: 'Tomato 2',
        122: 'Tomato 3',
        123: 'Tomato 4',
        124: 'Tomato Cherry Red',
        125: 'Tomato Heart',
        126: 'Tomato Maroon',
        127: 'Tomato Yellow',
        128: 'Tomato not Ripened',
        129: 'Walnut',
        130: 'Watermelon'
    }

    return class_names