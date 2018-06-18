import pandas as pd
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Data prepration
data = pd.read_csv('Bank_Data.csv')
x = data.iloc[:,3:13].values
y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])
# gender
labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])

# create dummy variables for country x x x
onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

#Splitting the dataset for Training and Testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Applying feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform((x_test))

#First hidden layer
classifier = Sequential()
classifier.add(Dense(output_dim=6,init="uniform",activation='relu',input_dim=11))
classifier.add(Dropout(rate=0.1))

#Second hidden layer
classifier.add(Dense(output_dim=6,init="uniform",activation='relu',input_dim=11))
classifier.add(Dropout(rate=0.1))

#Output layer
classifier.add(Dense(output_dim=1,init="uniform",activation='sigmoid'))

#Compile the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting classifier to the training data
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

#Save the model
classifier.save("nn_model.h5")

print('Model saved...')
print('All done.')

