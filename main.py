import os
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

uniq_labels = sorted(os.listdir('asl_dataset'))
print(uniq_labels)

def load_images(directory):
    image_dict = {}
    for idx, label in enumerate(uniq_labels):
        image_dict[label] = []
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (64, 64))
            image_dict[label].append(image)
    return image_dict

directory = 'asl_dataset'
image_dict = load_images(directory)

images = []
labels = []

for label, img_list in image_dict.items():
    images.extend(img_list)
    labels.extend([label] * len(img_list))

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# ---

import matplotlib.pyplot as plt

def display_images(images, labels, num_images=16):
    expected_labels = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]
    label_to_image = {}

    # Normalize label format
    labels = [str(label).upper() for label in labels]

    for img, lbl in zip(images, labels):
        if lbl in expected_labels and lbl not in label_to_image:
            label_to_image[lbl] = img
        if len(label_to_image) == len(expected_labels):
            break

    # Plotting
    plt.figure(figsize=(16, 10))
    for idx, lbl in enumerate(expected_labels):
        plt.subplot(6, 6, idx + 1)
        if lbl in label_to_image:
            img = label_to_image[lbl]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
        else:
            plt.text(0.5, 0.5, 'Missing', fontsize=12, ha='center')
        plt.title(lbl)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

print("Train images:")
display_images(X_train, y_train)
print("Test images:")
display_images(X_test, y_test)

# ---

import keras
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

all_labels = np.concatenate([y_train, y_test])
label_encoder.fit(all_labels)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(y_train[0])
print(len(y_train[0]))

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

# ---

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = 7, padding = 'same', activation = 'relu',
                 input_shape = (64, 64, 3)))
model.add(Conv2D(filters = 64, kernel_size = 7, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 128 , kernel_size = 7, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128 , kernel_size = 7, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 256 , kernel_size = 7, padding = 'same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(36, activation='softmax'))

model.summary()

# ---

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = model.fit(X_train, y_train, epochs = 10, batch_size = 64)

score = model.evaluate(x = X_test, y = y_test, verbose = 0)
print('Accuracy for test images:', round(score[1]*100, 3), '%')

# ---

from keras.models import load_model
model = load_model("asl_cnn_model.keras")

# ---

def plot_confusion_matrix(y, y_pred):
    y = np.argmax(y, axis = 1)
    y_pred = np.argmax(y_pred, axis = 1)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize = (24, 20))
    ax = plt.subplot()
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Purples)
    plt.colorbar()
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(uniq_labels))
    plt.xticks(tick_marks, uniq_labels, rotation=45)
    plt.yticks(tick_marks, uniq_labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    limit = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment = "center",color = "white" if cm[i, j] > limit else "black")
    plt.show()

from sklearn.metrics import confusion_matrix
import itertools

y_test_pred = model.predict(X_test, batch_size = 64, verbose = 0)
plot_confusion_matrix(y_test, y_test_pred)

# ---

from keras.models import save_model

model.save_model("asl_cnn_model.keras")
