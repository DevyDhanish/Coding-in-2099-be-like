import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import cv2

dataset_dir = "dataset"

img_height = 180
img_width = 180
img_channel = 3
batch = 32

train_data_generator = ImageDataGenerator(
    rotation_range=10,
    shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    fill_mode='nearest',
    rescale=1.0/255,
)

traning_data = train_data_generator.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch,
    subset="training"
)

validation_data = train_data_generator.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch,
    subset="validation",
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_width, img_height, img_channel)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation="relu"),
    #Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(2, activation="softmax"),
])


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
model.fit(traning_data, epochs=10, validation_data=validation_data)

vid = cv2.VideoCapture(0)

while 1:
    ret, img = vid.read()

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Cam feed", grayscale)
    grayscale = cv2.merge([grayscale, grayscale, grayscale])
    grayscale = cv2.resize(grayscale, (img_width, img_height))
    grayscale = np.array(grayscale).reshape(1, img_width, img_height, img_channel)

    output = model.predict(grayscale)
    print(f"Model thinks - {output}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()

print("Sucessfull")