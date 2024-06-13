import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
    
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units=16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim=30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0001,clipvalue=0.5)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics=['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [10,30],
              'epochs':[50,100],
              'optimizer':['adam','sgd'],
              'loos':['binary_crossentropy', 'hinge'],
              'kernel_initializer':['random_uniform', 'normal'],
              'activation':['relu', 'tanh'],
              'neurons':[16,8]
    }
grid_search = GridSearchCV(estimator = classificador, param_grid = parametros, scoring = 'accuracy', cv =5)
grid_search = grid_search.fit.best_params_
melhor_precisao = grid_search.best_score_