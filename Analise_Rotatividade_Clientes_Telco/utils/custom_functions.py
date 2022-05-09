import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import warnings
from sklearn.pipeline import Pipeline

#Definindo funções auxiliares de plotagem e manipulação de dataframes

def SumFeatureState(df, feature):
    
    """Função que calcula o total de uma feature por estado
    Recebe: feature numérica do dataframe explore
    Retorna: dataframe com duas colunas (Estado e Soma Agregada)
    """
    
    df_sum_state = df.groupby("state")[feature].sum().reset_index().sort_values(by = feature, ascending = False)
    return df_sum_state



#def CountGroupState(df):
    
   # """ Função que calcula o numéro de clientes que fizeram ou não churn por estado 
   # Retorna: Dataframe com 3 colunas (Estado, Fez ou Não Churn e Contagem)
   # """
    
  #  df_state = df.groupby(["state","churn"]).count().iloc[:,0].reset_index().rename(columns = {"account_length":"count"})
  #  return df_state


def RotationPropState(df):
    
    """ Função que calcula a proporção de clientes que fizeram churn em cada estado
    Retorna: Dataframe com 5 colunas (Estado, Churn (todos = yes), Número de clientes 
    que fizeram churn por estado, Total de Clientes por estado e Taxa de Churn por estado)
    """
    
    stateChurn = df.groupby(["state","churn"]).count().iloc[:,0].reset_index()
    yes_stateChurn = stateChurn[stateChurn["churn"] == "yes"].rename(columns = {"account_length":"count"})
    lenState = df["state"].value_counts().reset_index().rename(columns={"index":"state", "state":"totalState"})
    stateProp = pd.merge(left=yes_stateChurn, right=lenState, on="state")
    stateProp["prop"] = ((stateProp["count"] / stateProp["totalState"])*100).round(2)
    stateProp = stateProp.sort_values(ascending=False, by= "prop")
    return stateProp


def BarplotState(data, y, y_label, title = None, bars_annotate = False):
    
    """ Função que plota um gráfico de barras com os estados 
    no eixo X e qualquer outra variável no eixo Y
    Recebe: data (Dataframe), y: feature numérica do df (string), 
    y_label: nome do eixo y no gráfico (string), title: Título do gráfico (string),
    bars_annotate: mostrar ou não totais em cada barra (bool)
    """
    
    plt.figure(figsize=(20,10))
    plt.title(title)

    ax = sns.barplot(data=data, x = "state", y = y, color="blue")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Estado")
    
    if bars_annotate == True:
        for p in ax.patches:ax.annotate(format(p.get_height(), '.0f'), 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha = 'center', va = 'center', xytext = (0, 9),
        textcoords = 'offset points')
        
    plt.grid(False)


def StackedBarplotState(df):
    
    """Função que plota um gráfico de barras stacked contendo o total de clientes por estado 
    e quantos deles realizaram o churn"""


    df_plot = df.sort_values(by = "totalState", ascending = False)

    plt.figure(figsize = (20,12))
    plt.title("Total de Clientes por Estado")

    bar1 = sns.barplot(data = df_plot, x  = "state", y = "totalState", color = "lightblue")
    bar2 = sns.barplot(data = df_plot, x = "state", y = "count", color = "darkblue")

    bar1.set_ylabel("Total de Clientes")
    bar1.set_xlabel("Estado")

    for p in bar1.patches:
        bar1.annotate(format(p.get_height(), '.0f'), 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha = 'center', va = 'center', 
        xytext = (0, 9), #9
        textcoords = 'offset points')

    top_bar = mpatches.Patch(color='darkblue', label='churn')
    bottom_bar = mpatches.Patch(color='lightblue', label='not churn')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.grid(False)
    plt.show()

def grouped_barplot(categories, bar1, bar2, label1, label2, title, pos = 0 ):
    
    """Função que plota um grouped barplot
    Recebe: categories: categorias do eixo X (lista de strings),
    bar1: valores numéricos da 1a barra (lista de num),
    bar2: valores numéricos da 2a barra (lista de num),
    label1: legenda da barra 1 (string),
    label2: legenda da barra 2 (string), 
    title: titulo do grafico (string),
    pos: posição da legenda (0 a 4)"""
    
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [round(i,4) for i in bar1], width, label = label1)
    rects2 = ax.bar(x + width/2, [round(i,4) for i in bar2], width, label = label2)

    ax.set_title(title)
    ax.set_xticks(x,categories)
    ax.legend(loc = pos)

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    
    plt.show()


def get_feature_names(column_transformer):
    """Função retirada ed https://johaupt.github.io/blog/columnTransformer_feature_names.html"""
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline: #sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == Pipeline: #sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names