import streamlit as st
from streamlit_extras.stateful_button import button
import tensorflow
from tensorflow import keras

def mnist_densely_connected():
    # loading
    if button("1. mnist.load_data()", key="button1"):
        (train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.mnist.load_data()
        st.text(train_images.shape)
        st.text(len(train_labels))
        st.text(train_images.dtype)

        # building DL
        if button("2. model = keras.Sequential", key="button2"):
            from tensorflow import keras    
            model = keras.Sequential([
                keras.layers.Dense(512, activation = 'relu'),
                keras.layers.Dense(10, activation = 'softmax')
            ])
            
            # compiling
            if button("3. compile", key="button3"):
                model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # reshaping
                if button("4. reshaping", key="button4"):
                    train_images = train_images.reshape((60000, 28 * 28))
                    train_images = train_images.astype("float32") / 255
                    test_images = test_images.reshape((10000, 28 * 28))
                    test_images = test_images.astype("float32") / 255

                    # Fitting
                    if button("5. fit", key="button5"):
                        st.write(model.fit(train_images, train_labels, epochs=5, batch_size=128))

                        if button("6. testing", key="button6"):
                            test_digits = test_images[0:10]
                            predictions = model.predict(test_digits)
                            st.write(predictions)
                            st.info(predictions[0].argmax())
                            st.success(predictions[0][predictions[0].argmax()])
                            st.text(test_labels[0])

                            if button("7. Evaluating", key="button7"):
                                test_loss, test_acc = model.evaluate(test_images, test_labels)
                                st.info(f"test_acc: {test_acc}")


def mnist_convnets(): 
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=32, kernel_size=5, activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)    
    st.text(model.summary())
    
    if button("Start Trainning? ", key="button1"):
        (train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.mnist.load_data()
        train_images = train_images.reshape((60000,28,28,1))
        train_images = train_images.astype('float32')/255
        
        test_images = test_images.reshape((10000,28,28,1))
        test_images = test_images.astype('float32')/255
        
        model.compile(optimizer='rmsprop',
                      loss = 'sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(train_images, train_labels, epochs=4, batch_size=64)
        
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        st.text(f"test accuracy: {test_acc: .3f}")
