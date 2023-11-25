#import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np

#importation des données
url = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/ufc-master-final.csv'
df = pd.read_csv(url)

url_preprocessed = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/UFC_dataprocessed.csv'
df_preprocessed = pd.read_csv(url_preprocessed)

shape_df=df.shape
shape_dr_processed=df_preprocessed.shape

st.title('Apprentissage automatique - prédictions UFC')

intro_text='''Faire les meilleures prédictions possibles  
            sur les combats UFC '''

st.markdown(intro_text)

on=st.toggle('appuyez pour voir les données traitées')

if on:
    st.header('Données traitées')
    st.dataframe(df_preprocessed)


else:
    st.header('Données brutes')
    st.dataframe(df)
