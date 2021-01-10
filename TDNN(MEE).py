from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, MaxPooling1D, Conv1D, Flatten
from keras.layers import Dropout, GaussianNoise
from sklearn.metrics import mean_squared_error
import math
import keras.backend as K


def plot_signals(x, y, x_label, y_label, limit=1000):
    plt.plot(np.arange(len(x))[:limit], x.ravel().ravel()[:limit], label=x_label)
    plt.plot(np.arange(len(y))[:limit], y.ravel().ravel()[:limit], label=y_label)
    plt.legend()
    plt.show()

def gaussianKernel(x1, x2, sigma=0.05*math.sqrt(2)):
    sim = 0
    sim = K.exp(- K.abs( (x1-x2)*(x1-x2) ) / float( 2*(sigma**2) ) )  
    return sim

train_data = np.array(loadmat('train_tau_30.mat')['X1'])
test_data = np.array(loadmat('test_tau_30.mat')['Y1'])
plot_signals(train_data, test_data, 'train_data', 'test_data')

noise = 0.95*np.random.normal(0,0.1,train_data.shape) + 0.05*np.random.normal(1,0.5,train_data.shape)
noise_train_data = train_data + noise
plot_signals(train_data, noise_train_data, 'train_data', 'noise_train_data')



def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))

def mee(y_true, y_pred):
 value = gaussianKernel(y_pred,y_true)
 print('current loss',value)
 loss = K.sum(value)
 print('sum loss', loss)
 return -loss


window_size = 10
train_X, train_Y = create_dataset(train_data, window_size)
test_X, test_Y = create_dataset(test_data, window_size)
train_X = np.expand_dims(train_X, axis=2)
test_X = np.expand_dims(test_X, axis=2)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(window_size,1)))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss=mee, optimizer='adam')
epochs = 10
batch_size = 1
mse_model= model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size)

def predict_and_score(model, X, Y):
    pred = model.predict(X)
    score = math.sqrt(mean_squared_error(Y, pred))
    return(score, pred)

rmse_train, train_predict = predict_and_score(model, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)

plot_signals(train_data, train_predict, 'Training_data', 'Training_data_predict')
plot_signals(test_data, test_predict, 'Training_data', 'Testing_data_predict')

plt.plot(mse_model.history['loss'], label='MSE (training data)')
plt.plot(mee_model.history['loss'], label='MEE (training data)')
plt.title('MSE and MEE losses')
plt.ylabel('loss')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


#Training data score: 0.11 RMSE
Test data score: 0.02 RMSE