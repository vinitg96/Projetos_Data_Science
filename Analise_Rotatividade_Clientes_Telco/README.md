
Os dados utilizados são oriundos da competição [Customer Churn Prediction 2020](https://www.kaggle.com/competitions/customer-churn-prediction-2020/data). 
A solução final consistiu em um [App online](https://share.streamlit.io/vinitg96/app_analise_churn_telco/main/Churn_app.py) contruído com o streamlit. A aplicação contém um dashboard interativo com informações dos clientes, o modelo final treinado e apto a receber novos dados para realizar previsões além de um resumo com os insights gerados. O repositório pode ser acessado [aqui](https://github.com/vinitg96/App_Analise_Churn_Telco).


## Objetivos 

- Entender fatores que ocasionam o churn em empresas de telecomunicação
- Treinar um modelo de Machine Learning para prever a probabilidade de um cliente vir a abandonar a empresa
- Segmentar os clientes em grupos com base em seu comportamento
- Fornecer insights e sugestões, baseadas em dados, que auxiliem os tomadores de decisão a aumentar a retenção de clientes
- Realizar o deploy na forma de uma aplicação online


## Considerações Finais - Aspectos técnicos
- Apesar da Regressão Logistica ter sido o modelo que melhor generalizou as previsões em treino e validação, sua performance foi muito baixa em relação aos demais.
- A adição de novas features melhorou consideravelmente a performance de todos os modelos avaliados
- O modelo final (XGBoost) apresentou uma [precisão média (AP)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) de 90,6%, uma [precisão](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html?highlight=precision#sklearn.metrics.precision_score) de 100% e [revocação](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html?highlight=recall#sklearn.metrics.recall_score) de 86% para a classe 1 (clientes que realizaram o churn) nos dados de teste (dados não visto em nenhum momento pelo modelo).
- Dos 224 clientes que fizeram o churn (classe 1 ou positiva) nos dados de teste, 192 foram identificados corretamente (Verdadeiro Positivos) e 32 foram rotulados como clientes que permanecem (Falso Negativos)
- O número ideal de features foi 30.
- As tentativas de tratar o desbalanceamento nos dados (SMOTE e scale_pos_weight) não mostraram melhoria frente ao modelo padrão
- A melhor combinação de hiperparâmetros obtida foi n_estimators = 300, learning_rate = 0.6944, max_depth = 8 e subsample = 1
- As features que mais impactam as previsões do modelo foram: recharge_total, number_customer_services_calls e international_plan. Todas essas contribuem com o aumento da probabilidade de haver churn. 

## Considerações Finais - Aspectos de business
- A taxa de rotatividade média em todos os Estados dos EUA foi de 14,11% ± 5,02%.
- O Estado da Califórnia foi o com menor número de clientes e maior taxa de churn (27%). Um dos possíveis motivos é a elevada concorrência enfrenteda, tendo em vista que se trata do Estado mais populoso dos Estados Unidos.
-  Clientes que fazem o churn são aqueles que passam mais minutos em chamdas gerando mais cobranças, ou seja, justamente aqueles que agregam mais valor a empresa.
-  Clientes que abandonam a empresa tendem a ser mais ativo no período da manhã
-  Grande parte dos clientes que abandonam a companhia não tem nenhum dos planos ofertados (correio de voz e internacional).
-  Clientes tendem a passar bastante tempo em média (~100 meses) antes de realizar o churn
-  Clientes que fazem o churn realizam mais chamadas para atendimento ao cliente em relação aos clientes que permanecem.
-  Com base em seus hábitos de consumo, os clientes podem ser segmentados em quatro grupos. Chama atenção a diferença do grupo que contém os clientes com alta probabilidade de deixar a empresa para os demais. Isso mostra que esses clientes tem um comportamento peculiar que permite diferenciá-los.
