#import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np


st.title('Apprentissage automatique - prédictions UFC')

st.text('Notre projet consiste à faire les meilleures prédictions possibles sur des données de UFC')

st.header('Voici les données brutes:')
#importation des données
url = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/ufc-master-final.csv'
df = pd.read_csv(url)

st.dataframe(df)

