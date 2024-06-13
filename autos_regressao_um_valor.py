import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.compose import ColumnTransformer

base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

base['name'].value_counts()
base = base.drop('name', axis = 1)
base['seller'].value_counts()
base = base.drop('seller', axis = 1)
base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)

i1 = base.loc[base.price <= 10]
base.price.mean()
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]


# Usamos o pandas para preencher os valores nulos/atributos faltantes
# Com os atributos mais comuns no database
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() 
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() 
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() 
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() 
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() 

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder1 = LabelEncoder()
encoder2 = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
previsores = encoder2.fit_transform(previsores).toarray()
#previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
#previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
#previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
#previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
#previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
#previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
#previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

# 0 0 0 0
# 2 0 1 0
# 3 0 0 1
#onehotencoder = OneHotEncoder(categorical_features = [0,1,3,5,8,9,10])


regressor = Sequential()
regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
regressor.add(Dense(units = 158, activation = 'relu'))
regressor.add(Dense(units = 1, activation = 'linear'))
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsoes = regressor.predict(previsores)
preco_real.mean()
previsoes.mean()
























