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
shape_df_colonnes=shape_df[1]
shape_df_lignes=shape_df[0]

shape_df_processed=df_preprocessed.shape
shape_df_processed_colonnes=shape_df[1]
shape_df_processed_lignes=shape_df[0]

st.title('Apprentissage automatique - prédictions UFC')

intro_text='''Faire les meilleures prédictions possibles sur les combats UFC '''

st.header(intro_text)

on=st.toggle('appuyez pour voir les données traitées')
with st.container():
    if on:
            st.header('Données traitées')
            st.dataframe(df_preprocessed)
            st.write("Les données traitées ont ",shape_df_processed_lignes," lignes et ",shape_df_processed_colonnes," colonnes.")
    else:
            st.header('Données brutes')
            st.dataframe(df)
            st.write("Les données brutes ont ",shape_df_lignes," lignes et ",shape_df_colonnes," colonnes.")

Explanation_text_traitement_donnes='''Les principales modifications sont les suivantes:  
- Les données permettant d'dentifier une combattant directement (comme les noms) ont étés retirées  
- Les données en liens avec les cotes pour faire des paris furent retirées
- 
'''
st.markdown(Explanation_text_traitement_donnes)


