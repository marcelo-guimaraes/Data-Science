#Fraud detection using SVM
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier


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

X_train, X_teste, y_train, y_teste = train_test_split(features, classe, 
                                                    test_size = 0.3,
                                                    random_state = 0)

svm = SVC()
svm.fit(X_train, y_train)

previsoes = svm.predict(X_teste)

taxa_acerto = accuracy_score(y_teste, previsoes)

forest = ExtraTreesClassifier()
forest.fit(X_train, y_train)

importantes = forest.feature_importances_

X_train2 = X_train[:,0:5]
X_teste2 = X_teste[:,0:5]

svm.fit(X_train2, y_train)

previsoes2 = svm.predict(X_teste2)

taxa_acerto2 = accuracy_score(y_teste, previsoes2)



