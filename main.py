import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # Importa o escalonador padrão

data = pd.read_csv('./Data/final_data.csv')
print(data.head())

X = data.drop('Class', axis=1)  # Colunas a serem usadas como recursos, excluindo 'Class'
y = data['Class']  # Coluna 'Class' como o alvo

# Codificar variáveis categóricas usando one-hot encoding
X_encoded = pd.get_dummies(X, columns=['Company_Name', 'Designation', 'Location', 'Level', 'Involvement', 'Industry'], drop_first=True)

# Escalona os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Regressão Logística
lr = LogisticRegression(max_iter=1000)  # Aumente o número de iterações
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f'Acuracia da Regressão Logística: {lr_accuracy}')

# Árvore de Decisão
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f'Acurácia da Árvore de Decisão: {dt_accuracy}')

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'Acurácia da Random Forest: {rf_accuracy}')

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f'Acurácia da SVM: {svm_accuracy}')

# K-Nearest Neighbors (K-NN)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(f'Acurácia da KNN: {knn_accuracy}')
