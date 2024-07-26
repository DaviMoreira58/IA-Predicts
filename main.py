import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

source = os.path.abspath('.')
path = os.path.join(source, 'db')
db_clt = os.path.join(path, 'clientes.csv')

print(os.path.exists(db_clt))