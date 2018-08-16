from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt

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

	model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mae'])
	return model

'''
k = 4
#k-test
n_samples = len(x_train) // k
n_epochs = 100
all_scores = []
for i in range(k):
	print('Processing No.#', i)
	val_x = x_train[i*n_samples:(i + 1)*n_samples]
	val_y = y_train[i*n_samples:(i + 1)*n_samples]
	
	partial_x_train = np.concatenate([x_train[:i*n_samples], x_train[(i+1)*n_samples:]],axis=0)
	partial_y_train = np.concatenate([y_train[:i*n_samples], y_train[(i+1)*n_samples:]],axis=0)
	
	
	model = build_model()
	model.fit(partial_x_train, partial_y_train, epochs=n_epochs, batch_size=1, verbose=0)
	val_mse, val_mae = model.evaluate(val_x, val_y, verbose=0)
	all_scores.append(val_mae)
	
print(all_scores)
'''

'''
#Train 500 epochs and record each model preformance
k = 4
n_samples = len(x_train) // k
n_epochs = 500
all_mae_histories = []
for i in range(k):
	print('Processing No.#', i)
	#prepare the validation data: data from partition
	val_x = x_train[i*n_samples:(i+1)*n_samples]
	val_y = y_train[i*n_samples:(i+1)*n_samples]
	
	#prepare the trainig data: data from all other partitions
	partial_x_train = np.concatenate([x_train[:i*n_samples], x_train[(i+1)*n_samples:]], axis=0)
	partial_y_train = np.concatenate([y_train[:i*n_samples], y_train[(i+1)*n_samples:]], axis=0)
	
	#bulid the keras model
	model = build_model()
	#train hte model (in silent mode, verbose=0)
	history = model.fit(partial_x_train, partial_y_train, validation_data=(val_x, val_y), epochs=n_epochs, batch_size=1, verbose=0)
	mae_history = history.history['val_mean_absolute_error']
	all_mae_histories.append(mae_history)
	
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(n_epochs)]
'''

'''#plot
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
'''

model = build_model()
model.fit(x_train, y_train, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
print(test_mae_score, test_mse_score)	

	
