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

url_preprocessed = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/UFC_dataprocessed.csv'
df_preprocessed = pd.read_csv(url_preprocessed)

st.text('Seulement 69 colonnes ont étés gardés sur les 119 présentent originalement')
st.text('Les changements suivants ont étés faits:')

st.dataframe(df_preprocessed)

on=st.toggle('données brutes')

if on:
    st.dataframe(df)
else:
    st.dataframe(df_preprocessed)
