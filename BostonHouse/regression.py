from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import boston_housing
import numpy as np

#load datasets
'''
本数据集由StatLib库取得，由CMU维护。
每个样本都是1970s晚期波士顿郊区的不同位置，每条数据含有13个属性， (404+182)*13
目标值是该位置房子的房价中位数（千dollar） 102*13
'''
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#pretreatment
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

x_test -= mean
x_test /= std


def build_model():
	model = Sequential()
	model.add(Dense(64, input_shape=(x_train.shape[1],)))
	model.add(Activation('relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1))

	model.compile(optimizer='rmsprop', loss='mean_squared_error', metrices=['mae'])
	return model
	
#k-test
k = 4
n_samples = len(x_train)
n_epochs = 100
all_scores = []
for i in range(k):
	print('Processing No.#', i)
	val_x = x_train[i*n_samples:(i + 1)*n_samples]
	val_y = y_train[i*n_samples:(i + 1)*n_samples]
	
	partial_x_train = np.concatenate(
	[x_train[:i*n_samples], 
	x_train[(i+1)*n_samples:]],
	axis=0)
	partial_y_train = np.concatenate(
	[y_train[:i*n_samples], 
	y_train[(i+1)*n_samples:]],
	axis=0)
	
	#Train the model, iterating on the data in batches of 32 samples
	model = build_model()
	model.fit(partial_x_train, partial_y_train, epochs=n_epochs, batch_size=1, verbose=0)
	val_mse, val_mae = model.evaluate(val_x, val_y, verbose=0)
	all_scores.append(val_mae)
	
