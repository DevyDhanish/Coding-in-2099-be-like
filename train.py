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
TRANING : bool = True
PERFORM_VIDEO_TEST : bool = False

def getTrainingConfirmation():
    choice = input("Proceed (y/n) --> ")

    if choice == "y" or choice == "Y":
        return True
    
    elif choice == "n" or choice == "N":
        return False
    
    else:
        print("Invalid input")
        return getTrainingConfirmation()
    
getVideoConfirmation = getTrainingConfirmation

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
    Conv2D(64, (3,3), activation="relu", input_shape=(img_width, img_height, img_channel)),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(256, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(2, activation="softmax"),
])

print("Proceed with Traning")
TRANING = getTrainingConfirmation()

if(TRANING):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    model.fit(traning_data, epochs=5, validation_data=validation_data)
    model.save("model.h5")
else:
    model = tf.keras.models.load_model("model.h5")

print("Proceed with video test")
PERFORM_VIDEO_TEST = getVideoConfirmation()

lables = ["Violent", "Peace"]

if(PERFORM_VIDEO_TEST):
    vid = cv2.VideoCapture(0)

    while 1:
        ret, img = vid.read()
        cv2.imshow("Cam feed", img)
        img = cv2.resize(img, (img_width, img_width))
        img = np.array(img).reshape(1, img_width, img_height, img_channel)

        output = model.predict(img)
        print(f"Model thinks - {lables[np.argmax(output)]}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

print("Sucessfull")