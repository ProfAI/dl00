import numpy as np
import matplotlib.pyplot as plt

from scripts.utils import set_random_seed
from time import time

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import History
from keras.datasets import fashion_mnist

# Carichiamo il dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Convertiamo le immagini in vettori
X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)

# Normalizziamo il dataset
X_train = X_train/255
X_test = X_test/255

# Codifichiamo il target in 10 variabili di comodo
labels = ["T-shirt/top","Pantalone","Pullover","Vestito","Cappotto","Sandalo","Maglietta","Sneaker","Borsa","Stivaletto"]

y_train_dummy = to_categorical(y_train, 10)
y_test_dummy = to_categorical(y_test, 10)

# Impostiamo un seed comune
set_random_seed(0)

#Creiamo il modello
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

from keras import optimizers

sgd = optimizers.SGD(momentum=0.9, nesterov=True)

# Compiliamo il modello
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Addestriamo il modello cronometrando il tempo di addestramento
history = History()
start_at = time()
model.fit(X_train, y_train_dummy, epochs=100, batch_size=512, callbacks=[history])
exec_time = time()-start_at

print("Tempo di addestramento: %d minuti e %d secondi" % (exec_time/60, exec_time%60))

# Mostriamo il grafico della variazione della funzione di costo ad ogni epoca
plt.figure(figsize=(14,10))
plt.title("Mini Batch Gradient Descent com Momentum")
plt.xlabel("Epoca")
plt.ylabel("Log-Loss")
plt.plot(history.history['loss'])
