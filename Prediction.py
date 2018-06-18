import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

classifier = load_model("nn_model.h5")

# predict
y_pred = classifier.predict(x_test) #probabilty
y_pred = (y_pred > 0.5)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = 1.0*(cm[0,0] + cm[1,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print(cm)
print(accuracy)