import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

source = os.path.abspath('.')
path = os.path.join(source, 'db')
db_clt = os.path.join(path, 'clientes.csv')

table = pd.read_csv(db_clt)

encoder = LabelEncoder()
table['profissao'] = encoder.fit_transform(table['profissao'])

table['mix_credito'] = encoder.fit_transform(table['mix_credito'])

table['comportamento_pagamento'] = encoder.fit_transform(table['comportamento_pagamento'])

# x is who the AI ​​can use to predict
# y is who the AI ​​wants to predict

x = table.drop(columns=['score_credito', 'id_cliente'])
y = table['score_credito']

x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.3)


model_forest = RandomForestClassifier()
model_knn = KNeighborsClassifier()
