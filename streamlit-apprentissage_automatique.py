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
shape_df_processed_colonnes=shape_df_processed[1]
shape_df_processed_lignes=shape_df_processed[0]

st.title('Apprentissage automatique - prédictions UFC')

intro_text='''But: Faire les meilleures prédictions possibles sur les combats UFC '''

st.header(intro_text)

tab1,tab2 = st.tabs(['Données brutes', 'Données traitées'])

tab1.subheader("Données brutes")
tab1.dataframe(df)

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

st.divider()
Explanation_text_traitement_donnes='''Les principales modifications sont les suivantes:  
- Les données permettant d'dentifier une combattant directement (comme les noms) ont étés retirées  
- Les données en liens avec les cotes pour faire des paris furent retirées
- Les combats qui se sont terminés par des égalités furent retirés
- 
- 
'''
st.markdown(Explanation_text_traitement_donnes)

st.header('Entrainement des modèles')

Process_explnation_mrkdwn='''Voulant obtenir les meilleures performances, nous avons utilisé les modèles suivants.:  
1. Une cible fut établie, en utilisant la régle naive et en prédisant que tous les combats seraient gagnés par les combatants dans le coins rouge
2. Un MLP fut entrainé avec les réglages de base  
3. Un MLP fut entrainé en utilisant la recherche en grille  
4. Un MLP fut entrainé en utilisant la recherche aléatoire  
5. Un arbre de décision utilisant le boosting (XGBoost) fut entrainé  
6. Une regression linéaire utilisant l'analyse en compsantes principale fut entrainé  

Nous avons également fait des essaits en modifiant notre jeux de données, mais aucun différence notable de fut notée et donc les résultats ne seront pas présentés. '''

