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

with st.container():
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

st.write('le code utilisé pour monter les modèles et les résultats sont présentés ci-bas:')

with st.container():
    tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(['Règle Naive', 'MLP','Gridsearch','Randomsearch','XGBoost','Regression linéaire'])

    with tab3:
        plt.clf()
        cm=confusion_matrix(Y_test,Y_test_naive)
        confusion= sns.heatmap(cm, annot=True, fmt='g')
        # labels, title and ticks
        confusion.set_xlabel('Predicted winner');confusion.set_ylabel('True winner'); 
        confusion.set_title('Confusion Matrix (naive)'); 
        confusion.xaxis.set_ticklabels(['Blue', 'Red']); confusion.yaxis.set_ticklabels(['Blue', 'Red'])
        st.pyplot(confusion.get_figure())
    with tab4: #MLP de base
        st.subheader("Données traitées")
        st.dataframe(df_preprocessed)
        st.write("Les données traitées ont ",shape_df_processed_lignes," lignes et ",shape_df_processed_colonnes," colonnes.")
    with tab5: #gridsearch
        plt.clf()
        cm2=confusion_matrix(Y_test,Y_test_naive)
        confusion2= sns.heatmap(cm2, annot=True, fmt='g')
        # labels, title and ticks
        confusion2.set_xlabel('Predicted winner');confusion2.set_ylabel('True winner'); 
        confusion2.set_title('Confusion Matrix (naive)'); 
        confusion2.xaxis.set_ticklabels(['Blue', 'Red']); confusion2.yaxis.set_ticklabels(['Blue', 'Red'])
        st.pyplot(confusion.get_figure())
        # code utilisé
        code_grid='''param_grid={'hidden_layer_sizes': [10,20,50,100,120,150],
       'solver':['sgd','lbfgs'],
       'alpha':[0.0001,0.001,0.01,0.1,1],
       'batch_size':[256,512],
       'learning_rate_init':[0.001,0.01,0.1,0.2],
       'momentum':[0.1,0.3,0.6,0.9],
       'max_iter':[4000]}'''
        
        st.write("Voici les paramètres utilisés pour faire l'entrainement")
        st.code(code_grid,language='python')
    with tab6: #randomsearch
        plt.clf()
        cm2=confusion_matrix(Y_test,Y_test_naive)
        confusion2= sns.heatmap(cm2, annot=True, fmt='g')
        # labels, title and ticks
        confusion2.set_xlabel('Predicted winner');confusion2.set_ylabel('True winner'); 
        confusion2.set_title('Confusion Matrix (naive)'); 
        confusion2.xaxis.set_ticklabels(['Blue', 'Red']); confusion2.yaxis.set_ticklabels(['Blue', 'Red'])
        st.pyplot(confusion.get_figure())

        st.write("une fonction fut créée pour générer des chiffres entiers aléatoirement et une autre pour générer des chiffres à décimale")
        col1, col2= st.columns(2)

        with col1:
            code_getthatint='''random.seed(12)
def getthatint(lower=1,upper=1000,count=30):
random.seed(12)
getthatlist=[]
for i in range(count):
x = random.randint(lower,upper)
getthatlist.append(x)
return getthatlist'''
            st.code(code_getthatint,language='python')
        
        with col2:
            code_getthatfloat='''random.seed(12)
def getthatfloat(lower=0.0001,upper=1,count=30):
random.seed(12)
getthatlist=[]
for i in range(count):
x = round(random.uniform(lower,upper),4)
getthatlist.append(x)
return getthatlist'''
            st.code(code_getthatfloat,language='python')
    with tab7:
        st.subheader("Données brutes")
    with tab8:
        st.subheader("Données brutes")