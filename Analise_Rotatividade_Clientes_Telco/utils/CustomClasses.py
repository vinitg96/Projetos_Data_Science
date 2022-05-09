from sklearn.pipeline import make_pipeline
import time
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import cross_val_predict
import numpy as np
import matplotlib.pyplot as plt
from utils.custom_functions import grouped_barplot




class evaluate_models:
    
    """Classe para comparar diferentes modelos utilizando a área sob 
    a curva precision-recall como métrica
    """
    
    def __init__(self, X_train, y_train, X_val, y_val):
        """Inicializando classe com atributos do tipo Dataframe"""
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
  
    def get_score(self, model_names, model_instances, transformer, folds):
        """ Função que avalia diferentes modelos nos dados de treino (validação cruzada) e
        nos dados de validação
        """
        self.model_names = model_names
        self.clas_reports = []
        self.train_scores = []
        self.val_scores = []

        start_time = time.time()

        for label, model in zip(self.model_names, model_instances):
        
            final_pipe = make_pipeline(transformer, model)
            probs_train = cross_val_predict(final_pipe, self.X_train, self.y_train, cv=folds, method="predict_proba", n_jobs=-1)[:,1]
            train_score = average_precision_score(self.y_train, probs_train)
            self.train_scores.append(train_score)
    
            final_pipe.fit(self.X_train,self.y_train)
            val_score = average_precision_score(self.y_val, final_pipe.predict_proba(self.X_val)[:,1])
            self.val_scores.append(val_score)

            clas_report = classification_report(self.y_val, final_pipe.predict(self.X_val), output_dict=True)
            self.clas_reports.append(clas_report)
    

        print(f"Tempo de execução: {(time.time() - start_time):.4f} segundos ---")

    def plot_scores(self):
        """ Plota em um gráfico de barras os resultados de AP em treino e e validação"""

        x = np.arange(len(self.model_names))
        width = 0.35

        fig,ax = plt.subplots()
        rects1 = ax.bar(x - width/2, [round(i,4) for i in self.train_scores], width, label='Train')
        rects2 = ax.bar(x + width/2, [round(i,4) for i in self.val_scores], width, label='Validation')

        ax.set_ylabel('Precisão Média (AP)')
        ax.set_title('Precisão Média em Treino (10 folds) e Validação')
        ax.set_xticks(x, self.model_names)
        ax.legend(loc = 2)

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.show()
  
    def show_class_report(self):
        """ Função que plota um gráfico de barras com valores de recall e precision 
        para classe 1 (churn)
        """

        recalls = []
        precisions = []
        for i in range(len(self.model_names)):
            recalls.append(self.clas_reports[i]["1"]["recall"])
            precisions.append(self.clas_reports[i]["1"]["precision"])

        grouped_barplot(self.model_names, recalls,precisions,"Recall","Precision","Precisão e Recall nos dados de validação", 2)