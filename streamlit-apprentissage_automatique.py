import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score,confusion_matrix,ConfusionMatrixDisplay, precision_score
from matplotlib import pyplot as plt


#importation des données
url = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/ufc-master-final.csv'
df = pd.read_csv(url)

url_preprocessed = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/UFC_dataprocessed.csv'
df_preprocessed = pd.read_csv(url_preprocessed)

url_Naif = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/Y_test_naive.csv'
Y_test_naive = pd.read_csv(url_Naif)

url_Y_test = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/Y_test.csv'
Y_test = pd.read_csv(url_Y_test)

url_df_correlation = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/df_correlation.csv'
df_correlation = pd.read_csv(url_df_correlation)

shape_df=df.shape
shape_df_colonnes=shape_df[1]
shape_df_lignes=shape_df[0]

shape_df_processed=df_preprocessed.shape
shape_df_processed_colonnes=shape_df_processed[1]
shape_df_processed_lignes=shape_df_processed[0]

st.title('Apprentissage automatique - prédictions UFC')

intro_text='''But: Faire les meilleures prédictions possibles sur les combats UFC '''

st.header(intro_text)

with st.container():
    tab1,tab2 = st.tabs(['Données brutes', 'Données traitées'])

    with tab1:
        st.subheader("Données brutes")
        st.dataframe(df)
        st.write("Les données brutes ont ",shape_df_lignes," lignes et ",shape_df_colonnes," colonnes.")
    with tab2:
        st.subheader("Données traitées")
        st.dataframe(df_preprocessed)
        st.write("Les données traitées ont ",shape_df_processed_lignes," lignes et ",shape_df_processed_colonnes," colonnes.")
        st.write("elles furent traitées suite à l'analyse exploratoire")

st.divider()
Explanation_text_traitement_donnes='''Les principales modifications sont les suivantes:  
- Les données permettant d'dentifier une combattant directement (comme les noms) ont étés retirées  
- Les données en liens avec les cotes pour faire des paris furent retirées
- Les combats qui se sont terminés par des égalités furent retirés
- 
- 
'''

st.markdown(Explanation_text_traitement_donnes)

st.header('Analyse exploratoire')
st.write('La corrélation entre les variables a été obtneu pour comprendre les données et savoir si certaines variables devraient êtres priorisées')

correlation=sns.heatmap(df_correlation.corr(),cmap='coolwarm')
correlation.set_title('Corrélation entre les variables')
st.pyplot(correlation.get_figure())

st.write("Le paquet de visualisation Sweetviz a également été utilisé pour faire une analyse exploratoire des données")
st.write("Les colonnes à garder et les transformations de données ont étés décidés suite à l'analyse exploratoire")

st.header('Entrainement des modèles')

Process_explnation_mrkdwn='''Voulant obtenir les meilleures performances, nous avons utilisé fait les étapes suivantes:  
1. Une cible fut établie, en utilisant la régle naive et en prédisant que tous les combats seraient gagnés par les combatants dans le coins rouge
2. Un MLP fut entrainé avec les réglages de base  
3. Un MLP fut entrainé en utilisant la recherche en grille  
4. Un MLP fut entrainé en utilisant la recherche aléatoire  
5. Un arbre de décision utilisant le boosting (XGBoost) fut entrainé  
6. Une regression linéaire utilisant l'analyse en compsantes principale fut entrainé  

Nous avons également fait des essaits en modifiant notre jeux de données, mais aucun différence notable de fut notée et donc les résultats ne seront pas présentés. '''

st.markdown(Process_explnation_mrkdwn)

st.text('le code utilisé pour monter les modèles et les résultats sont présentés ci-bas:')

cm=confusion_matrix(Y_test,Y_test_naive)
ax= sns.heatmap(cm, annot=True, fmt='g')
# labels, title and ticks
ax.set_xlabel('Predicted winner');ax.set_ylabel('True winner'); 
ax.set_title('Confusion Matrix (naive)'); 
ax.xaxis.set_ticklabels(['Blue', 'Red']); ax.yaxis.set_ticklabels(['Blue', 'Red'])

with st.container():
    tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(['Règle Naive', 'MLP','Gridsearch','Randomsearch','XGBoost','Regression linéaire'])

    with tab3:
        st.pyplot(ax.get_figure())
    with tab4:
        st.subheader("Données traitées")
        st.dataframe(df_preprocessed)
        st.write("Les données traitées ont ",shape_df_processed_lignes," lignes et ",shape_df_processed_colonnes," colonnes.")
    with tab5:
        st.subheader("Données brutes")
    with tab6:
        st.subheader("Données brutes")
    with tab7:
        st.subheader("Données brutes")
    with tab8:
        st.subheader("Données brutes")