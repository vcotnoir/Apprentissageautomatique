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

url_preprocessed = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/UFC_dataprocessed.csv'
df_preprocessed = pd.read_csv(url_preprocessed)

st.text('Seulement 69 colonnes ont étés gardés sur les 119 présentent originalement')
st.text('Les changements suivants ont étés faits:') 

on=st.toggle('appuyez pour voir les données traitées')

if on:
    st.header('Données prétraitées')
    st.dataframe(df_preprocessed)
else:
    st.header('Données brutes')
    st.dataframe(df)
