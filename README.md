# Deteccao_Fraudes
'''
O programa detecao_fraudes.ipynb implementa um sistema de detecção automática de fraudes em transações financeiras.
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
