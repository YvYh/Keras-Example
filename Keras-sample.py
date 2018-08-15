from keras.models import Sequential
from keras.layers import Dense, Activation


"""

model = Sequential([
Dense(32, units=784),
Activation('relu'),
Dense(10),
Activation('softmax'),
])
"""

model = Sequential()
model.add(Dense(32, input_shape=(784, ))) #input_dim=784
model.add(Activation('relu'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',matrics=['accuracy'])


#Generate dummy data
import numpy as np
data = np.random.random((1000, 1000))
labels = np.random.randint(2, size=(1000, 1))

#Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

