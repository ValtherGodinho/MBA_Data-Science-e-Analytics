# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:37:41 2025

@author: Valther
"""
#Importação das Bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

#Carregar os Dados
dados = pd.read_parquet('exercicio.parquet')
print(dados.head())

#Preparação dos Dados
# Removendo a coluna categorizada 'idade_cat' pois vamos usar a idade original
X = dados.drop(columns=['inadimplencia', 'idade_cat'])
y = dados['inadimplencia']

# Separar em treino (80%) e teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

#Treinar o Modelo
# Criar e treinar a árvore de decisão
arvore = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
arvore.fit(X_treino, y_treino)

# Avaliação do Modelo
# Fazer previsões no conjunto de teste
y_pred = arvore.predict(X_teste)

# Matriz de Confusão
cm = confusion_matrix(y_teste, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Inadimplente', 'Inadimplente'], yticklabels=['Não Inadimplente', 'Inadimplente'])
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Relatório de Classificação
print(classification_report(y_teste, y_pred))

# Acurácia
acuracia = accuracy_score(y_teste, y_pred)
print(f'Acurácia: {acuracia:.2%}')

#Visualizar a Árvore
plt.figure(figsize=(20, 10))
plot_tree(arvore, feature_names=X.columns.tolist(), class_names=['Não Inadimplente', 'Inadimplente'], filled=True)
plt.show()
