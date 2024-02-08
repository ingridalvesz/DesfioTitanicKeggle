## importando as bibliotecas necessárias para realizar esse desafio 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

## Machine learning
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('train.csv')
train

test = pd.read_csv('test.csv')
test

train.info()

train.isnull().sum()

train.describe()

test.info()

test.isnull().sum()

test.describe()

PassengerId = test['PassengerId']
PassengerId 

#começo do dataframe de test

titanic_df = pd.concat([train, test], ignore_index=True)
titanic_df

train_index = len(train)
test_index = len(titanic_df) - len(test)

titanic_df.isnull().sum()

titanic_df.head(2)

df= pd.DataFrame()

##SURVIVED

titanic_df['Survived'].unique()

titanic_df['Survived'].isnull().sum()

titanic_df['Survived'].value_counts()

sns.countplot(data = titanic_df, x = 'Survived')

def titanic_func(data, column, count = True):
    print(f'Quantidade de valores únicos: {data[column].nunique()}')
    print(f'\nQuais são os valores únicos: {data[column].unique()}')
    print(f'\nQuantidade de valores nulos: {data[column].isnull().sum()}')
    print(f'\nQuantidade por opção: \n{data[column].value_counts()}')
    
    if count == True:
        sns.countplot(data = data, x = column, hue = 'Survived')
    else:
        sns.displot(data[column], kde = True)
    
    
titanic_func(titanic_df, 'Survived')

df['Survived'] = titanic_df['Survived']
df

##Pclass

titanic_func(titanic_df, 'Pclass')

df['Pclass'] = titanic_df['Pclass']
df

##Sex

titanic_df['Sex'] = titanic_df['Sex'].replace(['male', 'female'],[0,1])

titanic_func(titanic_df, 'Sex')

df['Sex'] = titanic_df['Sex']
df

##Age

titanic_func(titanic_df, 'Age', False)

df ['Age']= titanic_df['Age']
df

titanic_df['Age'].mean()

titanic_df = df.corr()
titanic_df

### maior correlação é a de classe 

# /// para abranger todoas as correlações
titanic_df = pd.concat([train, test], ignore_index=True)
titanic_df
# ////

titanic_df[titanic_df['Pclass'] == 1]['Age'].mean()

titanic_df[titanic_df['Pclass'] == 2]['Age'].mean()

titanic_df[titanic_df['Pclass'] == 3]['Age'].mean()

for i in sorted (titanic_df['Pclass']. unique()):
    print(f'Pessoas da {i}ª classe tem a média de idade de: {titanic_df[titanic_df["Pclass"] == i]["Age"].mean():.2f}' ) 

titanic_df['Age'].isnull().sum()

for i in titanic_df.index:
    if pd.isnull(titanic_df['Age'][i]): 
        if titanic_df['Pclass'][i] == 1:
            titanic_df['Age'][i] = round(titanic_df[titanic_df['Pclass'] == 1]['Age'].mean())
        elif titanic_df['Pclass'][i] == 2:
            titanic_df['Age'][i] = round(titanic_df[titanic_df['Pclass'] == 2]['Age'].mean())
        elif titanic_df['Pclass'][i] == 3:
            titanic_df['Age'][i] = round(titanic_df[titanic_df['Pclass'] == 3]['Age'].mean())
    else:
        continue

titanic_df[titanic_df['Pclass'] == 1]['Age'].isnull().sum()

titanic_df.isnull().sum()

df

##SibSp

titanic_func(titanic_df, 'SibSp')

df ['SibSp'] = titanic_df['SibSp']
df 

##Parch

titanic_func(titanic_df, 'Parch')

df ['Parch'] = titanic_df['Parch']
df 

##FamilySize

titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
titanic_df.head(3)

df ['FamilySize'] = titanic_df['FamilySize']
df 

##Fare

titanic_func(titanic_df, 'Fare', False)

titanic_df[titanic_df['Fare'].isnull()]

titanic_df[titanic_df['Pclass'] == 3]["Fare"].mean()

titanic_df['Fare'].fillna(titanic_df[titanic_df['Pclass'] == 3]["Fare"].mean(), inplace=True)

titanic_df.isnull().sum()

df ['Fare'] = titanic_df['Fare']
df 

##Embarked

titanic_func(titanic_df, 'Embarked')

titanic_df[titanic_df['Embarked'] == 'S']['Survived'].mean()
titanic_df[titanic_df['Embarked'] == 'S']['Pclass'].mean()

titanic_df[titanic_df['Embarked'] == 'C']['Survived'].mean()
titanic_df[titanic_df['Embarked'] == 'C']['Pclass'].mean()

titanic_df[titanic_df['Embarked'] == 'Q']['Survived'].mean()
titanic_df[titanic_df['Embarked'] == 'Q']['Pclass'].mean()

titanic_func[titanic_df['Embarked'].isnull()]

titanic_df['Embarked'].fillna("C", inplace = True)

titanic_df.isnull().sum()

df ['Embarked'] = titanic_df['Embarked']
df 

##Name

titanic_df['Name']


titanic_df['Title'] = titanic_df["Name"].apply(lambda name: name.split(',')[1].split('.')[0].strip())
titanic_df['Title']

titanic_df['Title'].nunique()

titanic_df['Title'].unique()

titanic_df['Title'].value_counts()

df ['Title'] = titanic_df['Title']
df 

##Finally

pclass = pd.get_dummies(df['Pclass'], prefix= "Pclass", drop_first=True)
title = pd.get_dummies(df['Title'], prefix= "Title", drop_first=True)
embarked = pd.get_dummies(df['Embarked'], prefix= "Embarked", drop_first=True)

titanic_concluido = pd.concat([df, pclass, title, embarked], axis = 1)
titanic_concluido

titanic_concluido.drop(['Pclass', 'Title', 'Embarked'], axis=1, inplace=True)
titanic_concluido

train = titanic_concluido[:train_index].copy()
test = titanic_concluido[test_index:].copy()

train

train['Survived'] = train['Survived'].astype(int)
train

X = train.drop('Survived', axis=1)
y = train['Survived']

X_test = test.drop('Survived', axis=1)

####

def func_acuracia(algoritmo, X_train, Y_train, vc):
    modelo = algoritmo.fit(X_train, Y_train)
    acuracia = round(modelo.score(X_train, Y_train) * 100, 2)

    train_pred = model_selection.cross_val_predict(algoritmo, X_train, Y_train, cv = vc, n_jobs = -1)
    acuracia_vc = round(metrics.accuracy_score(Y_train,train_pred ) * 100, 2)

    return acuracia, acuracia_vc


# Random Forest

acc_rf, acc_vc_rf = func_acuracia(RandomForestClassifier(), X, y, 10)

print(f"Acurácia: {acc_rf}")
print(f"Acurácia Validação Cruzada: {acc_vc_rf}")

#%%