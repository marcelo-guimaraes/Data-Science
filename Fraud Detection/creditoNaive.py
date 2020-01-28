import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

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

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

previsao = naive_bayes.predict(X_teste)
confusao = confusion_matrix(y_teste, previsao)
acuracia = accuracy_score(y_teste, previsao)

novo = pd.read_csv('NovoCredit.csv')
novo = novo.iloc[:,0:20].values

novo[:,0] = labelencoder.fit_transform(novo[:,0])
novo[:,2] = labelencoder.fit_transform(novo[:,2])
novo[:,3] = labelencoder.fit_transform(novo[:,3])
novo[:,5] = labelencoder.fit_transform(novo[:,5])
novo[:,6] = labelencoder.fit_transform(novo[:,6])
novo[:,8] = labelencoder.fit_transform(novo[:,8])
novo[:,9] = labelencoder.fit_transform(novo[:,9])
novo[:,11] = labelencoder.fit_transform(novo[:,11])
novo[:,13] = labelencoder.fit_transform(novo[:,13])
novo[:,14] = labelencoder.fit_transform(novo[:,14])
novo[:,16] = labelencoder.fit_transform(novo[:,16])
novo[:,18] = labelencoder.fit_transform(novo[:,18])
novo[:,19] = labelencoder.fit_transform(novo[:,19])

previsao_novo = naive_bayes.predict(novo)







