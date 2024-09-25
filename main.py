import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# train model
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print(f'Test accuracy: {test_accuracy}, Test loss: {test_loss}')


# model.save('hand_written.keras')





# test model
model = tf.keras.models.load_model('hand_written.keras')

image_number = 1
while os.path.isfile(f"numbers/{image_number}.png"):
    try:
        img = cv2.imread(f"new_numbers/{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number += 1