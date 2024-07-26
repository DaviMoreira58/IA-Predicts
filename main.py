import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

source = os.path.abspath('.')
path = os.path.join(source, 'db')
db_clt = os.path.join(path, 'clientes.csv')

table = pd.read_csv(db_clt)

encoder = LabelEncoder()
table['profissao'] = encoder.fit_transform(table['profissao'])

table['mix_credito'] = encoder.fit_transform(table['mix_credito'])

table['comportamento_pagamento'] = encoder.fit_transform(table['comportamento_pagamento'])

