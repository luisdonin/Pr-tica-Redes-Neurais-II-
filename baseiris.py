import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = to_categorical(classe, num_classes=3)


from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)



classificador = Sequential()
#camadas ocultas
classificador.add(Dense(units=4, activation='relu', input_dim=4))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax'))#problemas com mais de duas classes
classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)
