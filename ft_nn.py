from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("_%d-%m-%Y_%H-%M-%S") # dd/mm/YY H:M:S


# Import the data and/or model (2D version)
poly_density = 'pA.dat'
pA           = pd.read_csv(poly_density, delimiter = ' ',header=None, index_col = False)
pA.columns   = ['x_pos', 'y_pos', 'a_density']

pressure_field = 'wp.dat'
wp             = pd.read_csv(pressure_field, delimited = ' ', header=None, index_col = False)
wp.columns     = ['x_pos', 'y_pos', 'wp_field']

exchange_field = 'wx.dat'
wx             = pd.read_csv(exchange_field, delimited = ' ', header=None, index_col = False)
wx.columns     = ['x_pos', 'y_pos', 'wp_field']

# Split predictors and response and scale data
X = pA[['a_density']].values
Y = 
X = potential_data[['x_pos', 'z_pos']].values
y = potential_data['electric_potential'].values

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(X)
scaler_y.fit(y)
xscale = scaler_x.transform(X)
yscale = scaler_y.transform(y)

# Model achitecture
number_of_layers = 5 # Number of hidden layers minus one
numNeuron = 50 # Number of neurons per layer
      
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size = 0.01, random_state = 0)

# Neural Network Construction
model = Sequential()
model.add(Dense(numNeuron, input_dim = 1, activation = 'relu'))

NL = number_of_layers
model_arc = 'NL' + str(NL+1)  + '-NN' + str(numNeuron)

for layerrr in range(NL):
   model.add(Dense(numNeuron, activation = 'relu'))

# Output Layer
model.add(Dense(2, activation = 'linear'))

model.compile(loss = 'mse', optimizer =  'adam', metrics=['mean_squared_error'])

# Neural Network Execution
print(model.summary())
num_layers = len(model.layers)
print(model_arc)

def lr_scheduler(epoch, lr):
  decay_rate = 0.5
  decay_step = 100
  if epoch % decay_step == 0 and epoch:
     return lr * decay_rate
  return lr

callbacks = [
    keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
]

potential_model = model.fit(
   X_train, 
   y_train,
   validation_split=0.1, 
   epochs = 1000, 
   batch_size = 256,
   callbacks=callbacks,
)

# Evaluation on test set
predictions = model.predict(X_test)

y_test = y_test
y_pred = predictions

r_squared = r2_score(y_test, y_pred)
print('Testing r2: %.9f' % (r_squared))


# Plots

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot()
plt.loglog(potential_model.history['lr'])

plt.title('model mse')
#plt.xlim(left=1e0,right=1e3)
#plt.ylim(bottom=1e-6, top=1e0)
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
name = "learning_rate_" + model_arc + ".png"
fig.savefig(name, dpi=600)
plt.close()



# summarize history for accuracy
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot()
plt.loglog(potential_model.history['mean_squared_error'])
plt.loglog(potential_model.history['val_mean_squared_error'])
plt.title('Model mse')
plt.xlim(left=1e0,right=1e3)
plt.ylim(bottom=1e-7, top=0.5e0)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
name = "mse_" + model_arc + ".png"
fig.savefig(name, dpi=600)
plt.close()

# summarize history for loss
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot()
plt.loglog(potential_model.history['loss'])
plt.loglog(potential_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#name = "loss" + dt_string + ".png"
name = "loss_" + model_arc + ".png"
fig.savefig(name, dpi=600)
plt.close()


# Writting of biases and weights

df = pd.DataFrame(columns=['dummy'])

for layerNum, layer in enumerate(model.layers):
   df["Layer_{}".format(layerNum)] = layer.get_weights() # bias


B = df['Layer_0'][1].flatten() # bias 
W = df['Layer_0'][0].flatten() # weights 
W1 = np.append(B,W)

B = df['Layer_1'][1].flatten() # bias 
W = df['Layer_1'][0].flatten() # weights 
W2 = np.append(B,W)

B = df['Layer_2'][1].flatten() # bias 
W = df['Layer_2'][0].flatten() # weights 
W3 = np.append(B,W)

B = df['Layer_3'][1].flatten() # bias 
W = df['Layer_3'][0].flatten() # weights 
W4 = np.append(B,W)

B = df['Layer_4'][1].flatten() # bias 
W = df['Layer_4'][0].flatten() # weights 
W5 = np.append(B,W)

B = df['Layer_5'][1].flatten() # bias 
W = df['Layer_5'][0].flatten() # weights 
W6 = np.append(B,W)


with open('w1.txt', 'w') as file:
   for row in W1:
       np.savetxt(file, [row], fmt='%f')
 
with open('w2.txt', 'w') as file:
   for row in W2:
       np.savetxt(file, [row], fmt='%f')
 
with open('w3.txt', 'w') as file:
   for row in W3:
       np.savetxt(file, [row], fmt='%f')
       
with open('w4.txt', 'w') as file:
   for row in W3:
       np.savetxt(file, [row], fmt='%f')
       
with open('w5.txt', 'w') as file:
   for row in W3:
       np.savetxt(file, [row], fmt='%f')
       
with open('w6.txt', 'w') as file:
   for row in W3:
       np.savetxt(file, [row], fmt='%f')

