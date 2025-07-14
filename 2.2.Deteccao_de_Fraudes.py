'''
O programa detecao_fraudes.py implementa um sistema de detecção automática de fraudes em transações financeiras.
Ele realiza todo o processo, desde o carregamento dos dados até a previsão de possíveis fraudes em novas transações,
utilizando uma rede neural MLP (Multi-Layer Perceptron) da biblioteca scikit-learn.

🔍 Resumo do funcionamento do programa

1. Importação dos dados

Lê um arquivo CSV chamado fraudes.csv, que contém registros de transações com atributos como:

amount (valor da transação),

time (momento da transação),

location (origem da transação: online ou loja física),

is_fraud (1 para fraude, 0 para transação legítima),

e outros como transaction_id e customer_id.

2. Pré-processamento dos dados

Remove colunas inúteis (transaction_id, customer_id) que não ajudam na previsão.

Codifica a variável categórica location em formato numérico (one-hot encoding).

Separa os dados em:

X: variáveis preditoras,

y: variável alvo (is_fraud).

Normaliza os dados numéricos (amount, time) para padronizar a escala.

Divide os dados em treino e teste (80%/20%).

3. Treinamento da rede neural

Cria uma rede neural com 3 camadas ocultas (10, 8 e 10 neurônios).

Treina o modelo com os dados de treino.

4. Avaliação do modelo

Faz previsões no conjunto de teste.

Gera um relatório de classificação com métricas como precisão, recall e f1-score, para avaliar o desempenho do modelo.

5. Previsão em novos dados

Simula novas transações financeiras.

Pré-processa esses novos dados da mesma forma que os dados de treino:

Codificação de location,

Normalização de amount e time.

O modelo:

Prediz se cada nova transação é fraude (is_fraud_predicted),

Calcula a probabilidade de fraude (fraud_probability).

Mostra os resultados finais com probabilidade e decisão para cada transação.

✅ Resultado Final

O programa:

Treina um classificador de fraudes com base em dados históricos,

Avalia o desempenho do modelo,

Aplica esse modelo em novas transações para indicar:

Se provavelmente são fraudulentas,

E com que probabilidade isso acontece.

'''


#!/usr/bin/env python
# coding: utf-8

# -----------------------------------------------
# 2.2. Detecção de Fraudes com Rede Neural (MLP)
# -----------------------------------------------

# === Bibliotecas ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# === Importa e Visualiza os Dados ===
df = pd.read_csv("fraudes.csv")
print("Amostra dos dados:")
print(df.head())

# === Pré-Processamento ===

# Remove colunas que não agregam valor preditivo
df = df.drop(columns=['transaction_id', 'customer_id'])

# Codifica variáveis categóricas com one-hot encoding
df_encoded = pd.get_dummies(df, columns=['location'], drop_first=True)

# Separa os dados em variáveis preditoras (X) e alvo (y)
X = df_encoded.drop(columns=['is_fraud'])
y = df_encoded['is_fraud']

# Normaliza as colunas numéricas
scaler = StandardScaler()
X[['amount', 'time']] = scaler.fit_transform(X[['amount', 'time']])

# Divide em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDados de treino prontos para o modelo:")
print(X_train)

# === Treinamento ===
model = MLPClassifier(
    hidden_layer_sizes=(10, 8, 10),
    max_iter=500,
    random_state=42,
    learning_rate_init=0.01,
    activation='relu'
)

model.fit(X_train, y_train)
print("\nModelo treinado com sucesso.")

# === Avaliação ===
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=False)

print("\nRelatório de Classificação:")
print(report)

# === Previsão com Novos Dados ===

# Lista com novas transações fictícias
new_transactions = [
    {"amount": 3871.09, "time": 10, "location": "Loja Física"},
    {"amount": 12.56, "time": 12, "location": "Online"},
    {"amount": 4451.62, "time": 20, "location": "Online"},
    {"amount": 38.09, "time": 20, "location": "Loja Física"}
]

# Converte para DataFrame
df_new = pd.DataFrame(new_transactions)

# Aplica a mesma codificação de variáveis categóricas
df_new_encoded = pd.get_dummies(df_new, columns=['location'], drop_first=True)

# Garante que todas as colunas estejam no mesmo formato esperado pelo modelo
for col in X.columns:
    if col not in df_new_encoded.columns:
        df_new_encoded[col] = 0

# Reorganiza colunas para coincidir com os dados de treino
df_new_encoded = df_new_encoded[X.columns]

# Aplica a normalização com o mesmo scaler
df_new_encoded[['amount', 'time']] = scaler.transform(df_new_encoded[['amount', 'time']])

# Gera as previsões e probabilidades
predictions = model.predict(df_new_encoded)
probabilities = model.predict_proba(df_new_encoded)[:, 1]

# Adiciona as informações ao DataFrame original
df_new["fraud_probability"] = probabilities
df_new["is_fraud_predicted"] = predictions

# Exibe os resultados
print("\nPrevisões para Novas Transações:")
print(df_new[["amount", "time", "location", "fraud_probability", "is_fraud_predicted"]])


