# Avaliação 2

- Aluno: Patrick Pires
- Matrícula: 201810037211

## Enunciado

A partir da base de dados enviada e do estudo sobre pré-processamentos feitos na primeira parte do trabalho (corrigindo os problemas identificados na avaliação e a execução da seleção de variáveis para os que não avaliaram), apresentar novos resultados utilizando o modelo Decision Tree (DT), kNN e Naive Bayes e acrescente um outro algoritmo (pode ser ensemble ou redes neural, por exemplo) e discutir seus resultados, seguindo os processos para Data Mining.


Salve o nome do arquivo relatório com o sobrenome dos participantes em pdf e envie para o email karla.figueiredo@gmail.com

## Introdução

No trabalho anterior pude fazer uma análise exploratória na base de dados para a entender melhor. Só para lembrar, irei adicionar abaixo o resumo feito sobre a base de dados.


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atributo</th>
      <th>tipo</th>
      <th>valores faltantes</th>
      <th>outliers</th>
      <th>cardinalidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>quantitativo racional</td>
      <td>0</td>
      <td>não</td>
      <td>discreta</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>qualitativa nominal</td>
      <td>0</td>
      <td>n/a</td>
      <td>discreta (binária)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Job</td>
      <td>qualitativa ordinal</td>
      <td>0</td>
      <td>n/a</td>
      <td>discreta</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Housing</td>
      <td>qualitativa nominal</td>
      <td>0</td>
      <td>n/a</td>
      <td>discreta</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Saving accounts</td>
      <td>qualitativa ordinal</td>
      <td>183</td>
      <td>n/a</td>
      <td>discreta</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Checking account</td>
      <td>qualitativa ordinal</td>
      <td>394</td>
      <td>n/a</td>
      <td>discreta</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Credit amount</td>
      <td>quantitativo racional</td>
      <td>0</td>
      <td>não</td>
      <td>contínua</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Duration</td>
      <td>quantitativo racional</td>
      <td>0</td>
      <td>não</td>
      <td>discreta</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Purpose</td>
      <td>qualitativa nominal</td>
      <td>0</td>
      <td>n/a</td>
      <td>discreta</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Risk</td>
      <td>qualitativa nominal</td>
      <td>0</td>
      <td>n/a</td>
      <td>discreta (binária)</td>
    </tr>
  </tbody>
</table>
</div>



## Avaliação dos modelos

Irei avaliar 4 modelos: Árvore de Decisão, KNN, Naive Bayes e Redes Neurais. A avaliação será feita da seguinte maneira:
- Avaliação dos modelos mais simples entre si, i.e., do jeito que o `sklearn` os fornece.
- Avaliação dos modelos com melhores hiperparâmetros. Realizando uma busca em grade (`GridSearchCV`).
- Avaliação dos modelos com os melhores hiperparâmetros e preenchimento de "missing values" com a moda.
    - Irei optar pela estratégia de preencher os "missing values" e não remover linhas pelo mesmo aspecto discutido no trabalho anterior: há poucos registros e, além disso, ao remover linhas com registros faltantes, o desbalanceamento entre as classes acaba sendo perdido e o modelo não aprenderá de forma a refletir a realidade dos dados e sua distribuição.

### Critério de Avaliação

Quero que o modelo acerte mais na classificação de pessoas que são más (`bad`) devedoras. Isso para garantir que o modelo não diga que uma pessoa é boa devedora, i.e., pagará seu empréstimo, quando a pessoa não é e não pagará. Dessa forma, diminui as chances de sair no prejuízo.
Claro que não posso deixar de observar também a classificação dos bons (`good`) devedores, pois se o modelo os classifica errado, é dinheiro que deixa de entrar.

## Definindo as funções a serem utilizadas


```python
def fill_with_mode(df, column):
    '''
    Preenche valores faltantes de uma coluna com a moda, de acordo com a proporção da coluna alvo ('Risk').
    '''

    goods_filter = df['Risk'] == 'good'
    bads_filter = df['Risk'] == 'bad'

    goods = df[goods_filter]
    bads = df[bads_filter]

    good_mode = goods[column].mode()[0]
    bad_mode = bads[column].mode()[0]

    df.loc[goods_filter, column] = df.loc[goods_filter, column].fillna(good_mode)
    df.loc[bads_filter, column] = df.loc[bads_filter, column].fillna(bad_mode)
    
    return df

def handle_missing(df, column, with_mode=False):
    '''
    Lida com o preenchimento de valores faltantes de uma dada coluna. Ou preenche com a moda, ou com o valor 'missing'.
    '''

    new_df = df.copy()

    if with_mode:
        return fill_with_mode(new_df, column)

    new_df[column] = new_df[column].fillna('missing')
    return new_df
```

### Funções de preparação dos dados de acordo com o modelo


```python

def dt_prepare(df, fill_missing_with_mode=False):
    new_df = df.copy()

    new_df = handle_missing(new_df, 'Saving accounts', with_mode=fill_missing_with_mode)
    new_df = handle_missing(new_df, 'Checking account', with_mode=fill_missing_with_mode)

    categorical_nominal_cols = resumo_df[resumo_df['tipo'] == 'qualitativa nominal']['atributo'].tolist()
    categorical_ordinal_cols = resumo_df[resumo_df['tipo'] == 'qualitativa ordinal']['atributo'].tolist()

    categorical_nominal_cols.remove('Risk')  # 'Risk' is the target variable

    # handle using one-hot encoding for nominal categorical variables
    new_df = pd.get_dummies(new_df, columns=categorical_nominal_cols)

    # handle using ordinal encoding for ordinal categorical variables
    ordinal_encoder = OrdinalEncoder()
    new_df[categorical_ordinal_cols] = ordinal_encoder.fit_transform(new_df[categorical_ordinal_cols])

    return new_df

def knn_prepare(df, fill_missing_with_mode=False):
    new_df = df.copy()

    # preparacao de dados para a arvore de decisao
    new_df = handle_missing(new_df, 'Saving accounts', with_mode=fill_missing_with_mode)
    new_df = handle_missing(new_df, 'Checking account', with_mode=fill_missing_with_mode)

    categorical_nominal_cols = resumo_df[resumo_df['tipo'] == 'qualitativa nominal']['atributo'].tolist()
    categorical_ordinal_cols = resumo_df[resumo_df['tipo'] == 'qualitativa ordinal']['atributo'].tolist()
    numerical_cols = resumo_df[resumo_df['tipo'].str.contains('quantitativo')]['atributo'].tolist()

    categorical_nominal_cols.remove('Risk')  # 'Risk' is the target variable

    # handle using one-hot encoding for nominal categorical variables
    new_df = pd.get_dummies(new_df, columns=categorical_nominal_cols)

    # handle using ordinal encoding for ordinal categorical variables
    ordinal_encoder = OrdinalEncoder()
    new_df[categorical_ordinal_cols] = ordinal_encoder.fit_transform(new_df[categorical_ordinal_cols])

    # handle using standard scaling for numerical variables
    scaler = StandardScaler()
    new_df[numerical_cols] = scaler.fit_transform(new_df[numerical_cols])

    return new_df

def nb_prepare(df, fill_missing_with_mode=False):
    new_df = df.copy()

    new_df = handle_missing(new_df, 'Saving accounts', with_mode=fill_missing_with_mode)
    new_df = handle_missing(new_df, 'Checking account', with_mode=fill_missing_with_mode)

    categorical_nominal_cols = resumo_df[resumo_df['tipo'] == 'qualitativa nominal']['atributo'].tolist()
    categorical_ordinal_cols = resumo_df[resumo_df['tipo'] == 'qualitativa ordinal']['atributo'].tolist()
    continuous_cols = resumo_df[resumo_df['cardinalidade'] == 'contínua']['atributo'].tolist()

    # handle using label encoding for nominal categorical variables
    for col in categorical_nominal_cols:
        label_encoder = LabelEncoder()
        new_df[col] = label_encoder.fit_transform(new_df[col])

    # handle using ordinal encoding for ordinal categorical variables
    ordinal_encoder = OrdinalEncoder()
    new_df[categorical_ordinal_cols] = ordinal_encoder.fit_transform(new_df[categorical_ordinal_cols])
    new_df[categorical_ordinal_cols].astype(int)

    # handle using KBinsDiscretizer for continuous variables
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    new_df[continuous_cols] = discretizer.fit_transform(new_df[continuous_cols])

    return new_df

def nn_prepare(df, fill_missing_with_mode=False):
    new_df = df.copy()

    new_df = handle_missing(new_df, 'Saving accounts', with_mode=fill_missing_with_mode)
    new_df = handle_missing(new_df, 'Checking account', with_mode=fill_missing_with_mode)

    categorical_nominal_cols = resumo_df[resumo_df['tipo'] == 'qualitativa nominal']['atributo'].tolist()
    categorical_ordinal_cols = resumo_df[resumo_df['tipo'] == 'qualitativa ordinal']['atributo'].tolist()
    numerical_cols = resumo_df[resumo_df['tipo'].str.contains('quantitativo')]['atributo'].tolist()
    
    categorical_nominal_cols.remove('Risk')  # 'Risk' is the target variable
    
    # handle using one-hot encoding for nominal categorical variables
    new_df = pd.get_dummies(new_df, columns=categorical_nominal_cols)
    
    # handle using ordinal encoding for ordinal categorical variables
    ordinal_encoder = OrdinalEncoder()
    new_df[categorical_ordinal_cols] = ordinal_encoder.fit_transform(new_df[categorical_ordinal_cols])

    # handle using standard scaling for numerical variables
    scaler = StandardScaler()
    new_df[numerical_cols] = scaler.fit_transform(new_df[numerical_cols])
    
    return new_df
```

Definirei um dicionário onde posso acessar as funções de preparação dos dados de acordo com o modelo que quero utilizar. Isso para que a execução do experimento não fique repetitiva e eu possa facilmente trocar o modelo a ser utilizado. Na definição da função que executa experimentos, isso ficará mais claro.


```python
prepare = {
    'dt': dt_prepare,
    'knn': knn_prepare,
    'nb': nb_prepare,
    'nn': nn_prepare
}
```

Farei o mesmo com a definição dos hiperparâmetros de cada modelo, pelo mesmo motivo.


```python
dt_param_grid = {
    'max_depth': [2, 4, 8, 16, 32, 64, None],
    'min_samples_split': [2, 4, 8, 16, 32, 64],
    'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
}

nb_param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
}

nn_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)]
}

param_grids = {
    'dt': dt_param_grid,
    'knn': knn_param_grid,
    'nb': nb_param_grid,
    'nn': nn_param_grid
}
```

Abaixo, uma função que encontra o melhor modelo para o conjunto de dados fornecido, utilizando GridSearchCV.


```python
def find_best_model(model, df, scorer, model_acronym):
    '''
    Encontra o melhor modelo para o conjunto de dados fornecido, utilizando GridSearchCV.
    '''

    new_df = prepare[model_acronym](df)

    X = new_df.drop(columns=['Risk'], axis=1)
    y = new_df['Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = param_grids[model_acronym]
    dt_grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, n_jobs=-1)
    dt_grid_search.fit(X_train, y_train)

    return dt_grid_search.best_estimator_
```

Agora sim a função que executa o experimento de fato. Essa função foi escrita com o objetivo de poder executar cada um dos cenários que propus para serem avaliados, i.e.:

1. Com o modelo cru (do jeito que o sklearn fornece)
2. Com os melhores hiperparâmetros encontrados
3. Com os melhores hiperparâmetros encontrados e preenchendo os valores faltantes com a moda da coluna.

Para isso a função recebe alguns parâmetros:

- `use_best`: utilizada para dizer se o experimento deve utilizar o modelo com os melhores hiperparâmetros encontrados ou o modelo passado para o experimento (que será o modelo simples, padrão do `sklearn`).
- `pos_label`: utilizado para definir qual o rótulo a ser utilizado na métrica
- `fill_missing_with_mode`: utilizado para definir se os valores faltantes devem ser preenchidos com a moda da coluna.
- `model_acronym`: utilizado para definir qual o modelo a ser utilizado no experimento. Isso para que a função possa acessar o dicionário de funções de preparação dos dados e o dicionário de hiperparâmetros do modelo.

Dessa forma, para executar cada experimento que desejo basta passar os valores da seguinte maneira:

1. `use_best=False`, `fill_missing_with_mode=False`
2. `use_best=True`, `fill_missing_with_mode=False`
3. `use_best=True`, `fill_missing_with_mode=True`


```python

def execute_experiment(model, df, model_acronym, use_best=True, scorer=None, pos_label='bad', show_confusion_matrix=True, fill_missing_with_mode=False):
    new_df = prepare[model_acronym](df, fill_missing_with_mode=fill_missing_with_mode)

    X = new_df.drop(columns=['Risk'], axis=1)
    y = new_df['Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if use_best:
        model = find_best_model(model, df, scorer, model_acronym)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if show_confusion_matrix:
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title(f'{model_acronym} confusion matrix')
        plt.show()
    
    return recall_score(y_test, y_pred, pos_label=pos_label)
```

#### Experimento 1


```python
df = pd.read_csv('class_german_credit.csv')

# dicionário para armazenar os recalls de cada modelo em cada experimento
recalls = {
    'dt': [],
    'knn': [],
    'nb': [],
    'nn': []
}

experiments_defs = [
    {'model': DecisionTreeClassifier(random_state=42), 'df': df, 'model_acronym': 'dt', 'show_confusion_matrix': False},
    {'model': KNeighborsClassifier(), 'df': df, 'model_acronym': 'knn', 'show_confusion_matrix': False},
    {'model': GaussianNB(), 'df': df, 'model_acronym': 'nb', 'show_confusion_matrix': False, 'pos_label': 0},
    {'model': MLPClassifier(random_state=42), 'df': df, 'model_acronym': 'nn', 'show_confusion_matrix': False}
]

def execute_experiments(experiments_defs, recalls):
    for experiment in experiments_defs:
        model_acronym = experiment['model_acronym']
        recall = execute_experiment(**experiment)
        recall = f'{int(recall*100)}%'
        recalls[model_acronym].append(recall)

execute_experiments(experiments_defs, recalls)
```

#### Experimento 2


```python
scorer = make_scorer(recall_score, pos_label='bad')
scorer2 = make_scorer(recall_score, pos_label=0)

experiments_defs = [
    {'model': DecisionTreeClassifier(random_state=42), 'df': df, 'model_acronym': 'dt', 'use_best': True, 'scorer': scorer, 'show_confusion_matrix': False},
    {'model': KNeighborsClassifier(), 'df': df, 'model_acronym': 'knn', 'use_best': True, 'scorer': scorer, 'show_confusion_matrix': False},
    {'model': GaussianNB(), 'df': df, 'model_acronym': 'nb', 'pos_label': 0, 'use_best': True, 'scorer': scorer2, 'show_confusion_matrix': False},
    {'model': MLPClassifier(random_state=42), 'df': df, 'model_acronym': 'nn', 'use_best': True, 'scorer': scorer, 'show_confusion_matrix': False}
]

execute_experiments(experiments_defs, recalls)
```


#### Experimento 3


```python
experiments_defs = [
    {'model': DecisionTreeClassifier(random_state=42), 'df': df, 'model_acronym': 'dt', 'use_best': True, 'scorer': scorer, 'show_confusion_matrix': False, 'fill_missing_with_mode': True},
    {'model': KNeighborsClassifier(), 'df': df, 'model_acronym': 'knn', 'use_best': True, 'scorer': scorer, 'show_confusion_matrix': False, 'fill_missing_with_mode': True},
    {'model': GaussianNB(), 'df': df, 'model_acronym': 'nb', 'pos_label': 0, 'use_best': True, 'scorer': scorer2, 'show_confusion_matrix': False, 'fill_missing_with_mode': True},
    {'model': MLPClassifier(random_state=42), 'df': df, 'model_acronym': 'nn', 'use_best': True, 'scorer': scorer, 'show_confusion_matrix': False, 'fill_missing_with_mode': True}
]

execute_experiments(experiments_defs, recalls)
```


### Resultados


```python
pd.DataFrame(recalls)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dt</th>
      <th>knn</th>
      <th>nb</th>
      <th>nn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38%</td>
      <td>16%</td>
      <td>28%</td>
      <td>33%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44%</td>
      <td>27%</td>
      <td>28%</td>
      <td>33%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37%</td>
      <td>27%</td>
      <td>37%</td>
      <td>35%</td>
    </tr>
  </tbody>
</table>
</div>



O que pode ser observado para cada modelo:

- decision tree:
    - ao fazer a busca em grade: melhorou
    - ao preencher os valores faltantes com a moda: piorou
- knn:
    - busca em grade: melhorou
    - preencher os valores faltantes: não mudou
- naive bayes:
    - busca em grade: não mudou
    - preencher os valores faltantes: melhorou
- rede neural:
    - busca em grade: não mudou
    - preencher os valores faltantes: melhorou

Dentre todos esses modelos e estratégias, o que melhor se adequou ao que preciso foi a árvore de decisão com os melhores hiperparâmetros encontrados, mas sem o preenchimento dos valores faltantes com a moda. Talvez com uma estratégia de preenchimento diferente, o resultado fosse melhor.
