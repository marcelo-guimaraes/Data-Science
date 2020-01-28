#Fraud detecttion using RandomForest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


credito = pd.read_csv('Credit.csv')
features = credito.iloc[:,0:20].values
classe = credito.iloc[:,20].values

labelencoder = LabelEncoder()
features[:,0] = labelencoder.fit_transform(features[:,0])
features[:,2] = labelencoder.fit_transform(features[:,2])
features[:,3] = labelencoder.fit_transform(features[:,3])
features[:,5] = labelencoder.fit_transform(features[:,5])
features[:,6] = labelencoder.fit_transform(features[:,6])
features[:,8] = labelencoder.fit_transform(features[:,8])
features[:,9] = labelencoder.fit_transform(features[:,9])
features[:,11] = labelencoder.fit_transform(features[:,11])
features[:,13] = labelencoder.fit_transform(features[:,13])
features[:,14] = labelencoder.fit_transform(features[:,14])
features[:,16] = labelencoder.fit_transform(features[:,16])
features[:,18] = labelencoder.fit_transform(features[:,18])
features[:,19] = labelencoder.fit_transform(features[:,19])

X_train, X_test, y_train, y_test = train_test_split(features, classe, 
                                                    test_size = 0.3,
                                                    random_state = 0)

floresta = RandomForestClassifier(n_estimators = 100)
floresta.fit(X_train, y_train)

previsao = floresta.predict(X_test)

confusao = confusion_matrix(y_test, previsao)
taxa_acerto = accuracy_score(y_test, previsao)

taxa_acerto

