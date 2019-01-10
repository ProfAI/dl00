import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
from keras import optimizers

from keras.datasets import fashion_mnist

# Caricamento del dataset

labels = ["T-shirt/top","Pantalone","Pullover","Vestito","Cappotto","Sandalo","Camicia","Sneaker","Borsa","Stivaletto"]

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Encoding delle immagini

X_train = X_train.reshape(X_train.shape[0],28*28)
X_test = X_test.reshape(X_test.shape[0],28*28)

# Normalizzazione

X_train = X_train/255
X_test = X_test/255

# Encoding del target

num_classes=10

y_train_dummy = to_categorical(y_train, num_classes)
y_test_dummy = to_categorical(y_test, num_classes)

from keras.regularizers import l1_l2

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)))
model.add(Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train_dummy, epochs=100, batch_size=512)

metrics_train = model.evaluate(X_train, y_train_dummy, verbose=0)
metrics_test = model.evaluate(X_test, y_test_dummy, verbose=0)

print("Train Accuracy = %.4f - Train Loss = %.4f" % (metrics_train[1], metrics_train[0]))
print("Test Accuracy = %.4f - Test Loss = %.4f" % (metrics_test[1], metrics_test[0]))

from skimage import io

url = "https://cdn-tp1.mozu.com/23660-34943/cms/34943/files/c6334a30-6e18-4131-adc9-4065f8bc4516?max=300&_mzcb=_1532018938145"

image = io.imread(url)

plt.imshow(image)

from skimage.transform import resize
from skimage.color import rgb2gray

image_small = resize(image, (28,28))
image_gray = rgb2gray(image_small)

x = image_gray.reshape(1, 28*28)

x = 1. - x

plt.imshow(x.reshape(28,28), cmap="gray")

pred = model.predict_classes(x)
pred
labels[pred[0]]

proba = model.predict(x)
proba
