# Python para Classificação de Dados

Este código em Python realiza a classificação de dados utilizando diferentes algoritmos de aprendizado de máquina, como Regressão Logística, Árvore de Decisão, Random Forest, SVM (Support Vector Machine) e K-Nearest Neighbors (K-NN). O objetivo é avaliar a acurácia de cada modelo na tarefa de classificação.

## Bibliotecas Utilizadas

- `pandas` (importada como `pd`): Usada para manipulação de dados, leitura do conjunto de dados a partir de um arquivo CSV e criação de DataFrames.
- `sklearn` (Scikit-Learn): Uma biblioteca amplamente utilizada para aprendizado de máquina e mineração de dados.
  - `train_test_split`: Função para dividir o conjunto de dados em conjuntos de treinamento e teste.
  - `LogisticRegression`: Modelo de Regressão Logística para classificação.
  - `DecisionTreeClassifier`: Modelo de Árvore de Decisão para classificação.
  - `RandomForestClassifier`: Modelo de Random Forest para classificação.
  - `SVC` (Support Vector Classifier): Modelo SVM para classificação.
  - `KNeighborsClassifier`: Modelo K-NN para classificação.
  - `accuracy_score`: Métrica para avaliar a acurácia dos modelos.

## Fluxo de Execução

1. Leitura do Conjunto de Dados:
   - O código lê um conjunto de dados a partir do arquivo CSV `final_data.csv` e carrega-o em um DataFrame chamado `data`.

2. Preparação dos Dados:
   - As colunas a serem usadas como recursos são definidas na variável `X`, excluindo a coluna alvo `Class`.
   - A coluna alvo `Class` é definida na variável `y`.
   - Variáveis categóricas são codificadas usando one-hot encoding, o que cria colunas binárias para cada categoria.
   - Os dados são escalonados usando `StandardScaler` para padronização.

3. Divisão dos Dados:
   - O conjunto de dados é dividido em conjuntos de treinamento e teste usando `train_test_split`. 80% dos dados são usados para treinamento (`X_train`, `y_train`) e 20% para teste (`X_test`, `y_test`).

4. Modelos de Classificação e Avaliação:
   - Vários modelos de classificação são treinados nos dados de treinamento e avaliados nos dados de teste.
     - Regressão Logística (`LogisticRegression`)
     - Árvore de Decisão (`DecisionTreeClassifier`)
     - Random Forest (`RandomForestClassifier`)
     - SVM (`SVC`)
     - K-Nearest Neighbors (`KNeighborsClassifier`)
   - A acurácia é calculada para cada modelo usando `accuracy_score` e é impressa na saída.

5. Resultados:
   - As acurácias de cada modelo são exibidas na saída, permitindo a comparação do desempenho dos diferentes algoritmos de classificação.

