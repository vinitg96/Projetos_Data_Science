# Introdução
A satisfação do cliente é uma medida fundamental do sucesso de um banco. Clientes insatisfeitos cancelam seus serviços e raramente expressam sua insatisfação antes de sair. Clientes satisfeitos, por outro lado, se tornam defensores da marca!

O objetivo deste projeto é **prever se um cliente está satisfeito ou insatisfeito com sua experiência bancária**. Para tal, centenas de recursos anônimos de clientes do Banco Santander estão disponíveis no [Kaggle](https://www.kaggle.com/c/santander-customer-satisfaction). O modelo apresentado deve apresentar **no minímo 70% de acurácia**. 

# Desafios encontrados:
- Dados em diferentes escalas
- Classes da variável target desbalanceadas 
- Conjunto de dados com elevado número de atributos


# Soluções empregadas:
- *Oversampling* da classe minoritária com [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- Centralizar os dados com relação a média e colocá-los em unidades de desvio padrão com [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- Redução de dimensionalidade com [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- Avaliação da performance nos dados de treino com [*Cross Validation*](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html?highlight=cross_val_score#sklearn.model_selection.cross_val_score) ([*Confusion Matrix*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html?highlight=confusion%20matrix#sklearn.metrics.confusion_matrix), e [*Classification Report*](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report)) 
- Treinamento de um classificador com o algoritmo [*Random Forest*](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier)
- Definição de um [*pipeline*](https://scikit-learn.org/stable/modules/classes.html?highlight=pipeline#module-sklearn.pipeline) para pré-processamento dos dados de teste
- *Deploy* em uma página web estática utilizando o [GitHub Pages](https://pages.github.com/)

# Conclusões e perspectivas

O modelo criado apresentou uma acurácia média de 93% ± 1,5% nos 5 *folds* dos dados de treino além de um F1-score de 0.93. Avaliando os acertos entre as classes, a proporção foi similar (67.435 para satisfeitos e 68.384 para insatisfeitos) indicando que o balanceamento foi eficaz e o modelo não foi tendencioso. Ainda, o número de falsos positivos (5.557) foi maior que o de falsos negativos (4.628) refletindo uma proporção desejada considerando o contexto (é menos danoso ter clientes satisfeitos classificados como insatisfeitos do que o cntrário).

Logo, a análise pode contribuir para diminuir a evasão de clientes ao permitir identificar aqueles potencialmente instatisfeitos. Isso possibilitaria ao banco tomar medidas proativas para melhorar a felicidade dos clientes não satisfeitos antes que seja tarde de mais.

Por fim, futuramente é possível tentar melhorar o modelo com o ajuste de hiperparâmetros ou uso de outros algoritmos.

A análise completa se encontra em : https://vinitg96.github.io/Santander_Customer_Satisfaction/

Obs: não é necessário executar a célula de cross validation nem a de treinar o modelo, basta carregar os arquivos com os resultados nas células correspondentes. 





