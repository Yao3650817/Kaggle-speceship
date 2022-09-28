#import
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import metrics

spaceship=pd.read_csv("Desktop/kaggle/spaceship/train.csv")
spaceship_1=pd.read_csv("Desktop/kaggle/spaceship/test.csv")
#summary=pd.DataFrame(spaceship.describe())
#spaceship['Destination'].describe()
#spaceship['Cabin'].describe()
#spaceship['PassengerId'].describe()

#Drop column
spaceship.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)
spaceship_1.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)
print(spaceship.isnull().sum())
print(spaceship_1.isnull().sum())

#Input missing value
spaceship['Age'] = spaceship['Age'].fillna(spaceship['Age'].median())
spaceship['RoomService'] = spaceship['RoomService'].fillna(spaceship['RoomService'].median())
spaceship['FoodCourt'] = spaceship['FoodCourt'].fillna(spaceship['FoodCourt'].median())
spaceship['ShoppingMall'] = spaceship['ShoppingMall'].fillna(spaceship['ShoppingMall'].median())
spaceship['Spa'] = spaceship['Spa'].fillna(spaceship['Spa'].median())
spaceship['VRDeck'] = spaceship['VRDeck'].fillna(spaceship['VRDeck'].median())
spaceship_1['Age'] = spaceship_1['Age'].fillna(spaceship_1['Age'].median())
spaceship_1['RoomService'] = spaceship_1['RoomService'].fillna(spaceship_1['RoomService'].median())
spaceship_1['FoodCourt'] = spaceship_1['FoodCourt'].fillna(spaceship_1['FoodCourt'].median())
spaceship_1['ShoppingMall'] = spaceship_1['ShoppingMall'].fillna(spaceship_1['ShoppingMall'].median())
spaceship_1['Spa'] = spaceship_1['Spa'].fillna(spaceship_1['Spa'].median())
spaceship_1['VRDeck'] = spaceship_1['VRDeck'].fillna(spaceship_1['VRDeck'].median())

#Drop rest na
spaceship=spaceship.dropna()
spaceship_1=spaceship_1.dropna()


#corr=file.corr()
#sns.heatmap(spaceship.corr())
#plt.show()

#plotAge=plt.hist(spaceship['Age'])
#plotAge.show()
#plotRoom=plt.hist(spaceship['RoomService'],bins=20,range=(0,1000))
#plt.show(plotRoom)
#roomplot=spaceship.boxplot(column=['RoomService'])
#plt.show(roomplot)
#plotFood=plt.hist(spaceship['FoodCourt'])
#plotFood.show()

print (spaceship['CryoSleep'].dtypes)
spaceship.loc[:,"CryoSleep"] = spaceship.loc[:,"CryoSleep"].astype(int)
type(spaceship.iloc[8,1])

#dummy
spaceship = pd.get_dummies(spaceship, columns=['HomePlanet','Destination'])
spaceship_1 = pd.get_dummies(spaceship_1, columns=['HomePlanet','Destination'])
spaceship.drop(['HomePlanet_Earth','Destination_55 Cancri e'], axis=1, inplace=True)
spaceship_1.drop(['HomePlanet_Earth','Destination_55 Cancri e'], axis=1, inplace=True)

#add spending columns together
spaceship['spending']=(spaceship['RoomService']+spaceship['FoodCourt']+spaceship['ShoppingMall']
+spaceship['Spa']+spaceship['VRDeck'])

#drop individual spending
spaceship.drop(['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'],axis=1,inplace=True)

#do samething to test data
spaceship_1['spending']=(spaceship_1['RoomService']+spaceship_1['FoodCourt']+spaceship_1['ShoppingMall']
+spaceship_1['Spa']+spaceship_1['VRDeck'])

#drop individual spending
spaceship_1.drop(['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'],axis=1,inplace=True)

#split data
features = ['CryoSleep','Age','VIP','spending','HomePlanet_Mars',
'HomePlanet_Europa','Destination_PSO J318.5-22','Destination_TRAPPIST-1e']
x= spaceship.loc[:, features]
y = spaceship.loc[:, ['Transported']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

#logistic regression
logreg = LogisticRegression(max_iter=4000)
scores = cross_validate(logreg, x_train, y_train.values.ravel(), scoring='accuracy', cv=5)
print(scores['test_score'])

logreg.fit(x_train,y_train.values.ravel())

y_pred=logreg.predict(x_test)

# matrix
matrix = metrics.confusion_matrix(y_test, y_pred)
matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

predictions = logreg.predict(titanic_1)
output = pd.DataFrame({'PassengerId': titanic_1.PassengerId, 'Transported': predictions})
output.to_csv('spaceship upload.csv', index=False)
