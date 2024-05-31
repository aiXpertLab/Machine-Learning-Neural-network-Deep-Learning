import os, shutil, pathlib
import streamlit as st
from streamlit_extras.stateful_button import button
from os import listdir
from numpy import asarray, save
import numpy as np
from matplotlib import pyplot
from matplotlib.image import imread
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

train_dir = 'E:/models/kaggle/train'
train_dir_small ='E:/models/kaggle/small_dataset_train'
original_dir = pathlib.Path('E:/models/kaggle/train')
new_base_dir = pathlib.Path("E:/models/kaggle/small_dataset_train")

def data_pp():
    train_dataset = image_dataset_from_directory(
        new_base_dir / 'train',
        image_size = (180,180),
        batch_size = 32)
    
    validation_dataset = image_dataset_from_directory(
        new_base_dir / 'validation',
        image_size = (180, 180),
        batch_size = 32)

    test_dataset = image_dataset_from_directory(
        new_base_dir / 'test',
        image_size = (180, 180),
        batch_size = 32)
    
    for data_batch, labels_batch in train_dataset:
        st.text('hi')
        st.write('data batch shape: ', data_batch.shape)
        st.write('labels batch shape: ', labels_batch.shape)
        break
    
    return train_dataset, validation_dataset, test_dataset

def dc_preprocessing():
    for i in range(9):
        pyplot.subplot(330+1+i)
        filename = train_dir + '/cat.' + str(i) + '.jpg'
        image = imread(filename)
        pyplot.imshow(image)
    # pyplot.show()
    st.pyplot(pyplot.gcf())

    for i in range(9):
        pyplot.subplot(330+1+i)
        filename = train_dir + '/dog.' + str(i) + '.jpg'
        image = imread(filename)
        pyplot.imshow(image)
    # pyplot.show()
    st.pyplot(pyplot.gcf())

    st.write('''If we want to load all of the images into memory, we can estimate that it would require about
            12 gigabytes of RAM. That is 25,000 images with 200 × 200 × 3 pixels each, or 3,000,000,000
            32-bit pixel values. We could load all of the images, reshape them, and store them as a single
            NumPy array. This could fit into RAM on many modern machines, but not all, especially if
            you only have 8 gigabytes to work with. We can write custom code to load the images into
            memory and resize them as part of the loading process, then save them ready for modeling. The
            example below uses the Keras image processing API to load all 25,000 photos in the training
            dataset and reshapes them to 200 × 200 square photos. The label is also determined for each
            photo based on the filenames. A tuple of photos and labels is then saved.''')
    photos, labels = list(), list()
    # for file in listdir(train_dir):
    #     output = 0.0
        
    #     if file.startswith('cat'):   output = 1.0
    #     photo = load_img(train_dir + '/' + file, target_size=(200,200))
    #     photo = img_to_array(photo)
    #     photos.append(photo)
    #     labels.append(labels)
    # photos = asarray(photo)
    # labels = asarray(labels)
    # st.write(photos.shape, labels.shape)
    # save('E:/models/kaggle/dogs_vs_cats_photos.npy', photos)
    # save('E:/models/kaggle/dogs_vs_cats_labels.npy', labels)
            
def dc_small_dataset():

    def make_subset(subset_name, start_index, end_index):
        for category in ('cat', 'dog'):
            dir = new_base_dir / subset_name / category
            os.makedirs(dir)
            fnames = [f'{category}.{i}.jpg' for i in range(start_index, end_index)]
            for fname in fnames:
                shutil.copyfile(src = original_dir/fname, dst = dir/fname)
    #     make_subset("train", start_index=0, end_index=1000)
    #     make_subset("validation", start_index=1000, end_index=1500)
    #     make_subset("test", start_index=1500, end_index=2500)
    
    def model_definition():
        inputs = keras.Input(shape=(180, 180, 3))
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)    
        print(model.summary())
        
        model.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])
        
        return model
        
    def model_fit():
        model = model_definition()
        train_dataset, validation_dataset = data_pp()            
        
        history = model.fit(
            train_dataset,
            epochs=3,
            validation_data=validation_dataset,
            callbacks=callbacks)

        return history

    
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss")
]
    
    if button("Show Result", key="button1"): 
        train_dataset, validation_dataset, test_dataset = data_pp()
        test_model = keras.models.load_model("convnet_from_scratch.keras")
        test_loss, test_acc = test_model.evaluate(test_dataset)
        st.write(f"Test accuracy: {test_acc:.3f}")

        if button("Fit?", key="button2"): 
            
            
            history=model_fit()
            import matplotlib.pyplot as plt
            accuracy = history.history["accuracy"]
            val_accuracy = history.history["val_accuracy"]
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(1, len(accuracy) + 1)
            plt.plot(epochs, accuracy, "bo", label="Training accuracy")
            plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
            plt.title("Training and validation accuracy")
            plt.legend()
            plt.figure()
            plt.plot(epochs, loss, "bo", label="Training loss")
            plt.plot(epochs, val_loss, "b", label="Validation loss")
            plt.title("Training and validation loss")
            plt.legend()
            # plt.show()
            st.pyplot(plt.gcf())
            
            
def dc_vgg16_feature_extracting():
    conv_base = keras.applications.vgg16.VGG16(        weights = 'imagenet',        include_top = False,        input_shape = (180,180,3))
    st.write(conv_base.summary())
    
    def get_features(dataset):
        all_features, all_labels=[],[]
        for images, labels in dataset:
            preprocessed_images = keras.applications.vgg16.preprocess_input(images)
            features = conv_base.predict(preprocessed_images)
            all_features.append(features)
            all_labels.append(labels)
        return np.concatenate(all_features), np.concatenate(all_labels)
    
    train_dataset, validation_dataset, test_dataset = data_pp()
    
    train_features, train_labels = get_features(train_dataset)
    val_features, val_labels = get_features(validation_dataset)
    # test_features, test_labels = get_features(test_dataset)
    
    print(train_features.shape)
    
    inputs = keras.Input(shape= (5,5,512))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss="binary_crossentropy",        optimizer="rmsprop",        metrics=["accuracy"])
    callbacks = [        keras.callbacks.ModelCheckpoint(        filepath="feature_extraction.keras",        save_best_only=True,        monitor="val_loss")        ]
    history = model.fit(        train_features, train_labels,        epochs=20,        validation_data=(val_features, val_labels),        callbacks=callbacks)
    
    import matplotlib.pyplot as plt
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
    st.pyplot(plt.gcf())

def dc_vgg16_fine_tuning(): pass
