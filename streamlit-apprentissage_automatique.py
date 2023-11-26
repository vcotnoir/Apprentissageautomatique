import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score,confusion_matrix,ConfusionMatrixDisplay, precision_score
from matplotlib import pyplot as plt
import mpld3
import streamlit.components.v1 as components


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

url_predictions_random = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/clf_rand_pred.csv'
df_MLP_random = pd.read_csv(url_predictions_random)

url_preductions_XG = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/ufcboost_pred.csv'
df_xgboost = pd.read_csv(url_preductions_XG)

url_prediction_MLP = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/ufc_base_pred.csv'
df_MLP_base = pd.read_csv(url_prediction_MLP)

url_logistique = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/logis_predict.csv'
df_logistique = pd.read_csv(url_logistique)

url_eli5 = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/eli5_results.csv'
df_eli5 = pd.read_csv(url_eli5)

shape_df=df.shape
shape_df_colonnes=shape_df[1]
shape_df_lignes=shape_df[0]

shape_df_processed=df_preprocessed.shape
shape_df_processed_colonnes=shape_df_processed[1]
shape_df_processed_lignes=shape_df_processed[0]

st.title('Apprentissage automatique - prédictions UFC')

intro_text='''But: Faire les meilleures prédictions possibles sur les combats UFC '''

st.header(intro_text)

st.write("Les données brutes et les données traitées, utilisées pour faire l'entrainement des modèles, sont présentées ci-bas:")
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

st.header('Entrainement des modèles et résultats')

#Process_explnation_mrkdwn='''Voulant obtenir les meilleures performances, nous avons utilisé fait les étapes suivantes:  
#1. Une cible fut établie, en utilisant la régle naive et en prédisant que tous les combats seraient gagnés par les combatants dans le coins rouge
#2. Un MLP fut entrainé avec les réglages de base  
#3. Un MLP fut entrainé en utilisant la recherche en grille  
#4. Un MLP fut entrainé en utilisant la recherche aléatoire  
#5. Un arbre de décision utilisant le boosting (XGBoost) fut entrainé  
#6. Une regression linéaire utilisant l'analyse en compsantes principale fut entrainé  

# Nous avons également fait des essaits en modifiant notre jeux de données, mais aucun différence notable de fut notée et donc les résultats ne seront pas présentés. '''

# st.markdown(Process_explnation_mrkdwn)
explication = '''Les modèles suivants furent essayés, les résultats sont présentés initialement et le code par la suite.  
L'ordre de présentation représente l'ordre dans lesquels les modèles furent testés.'''
st.markdown(explication)

with st.container():
    naive,mlp,grid,random,xg,regression = st.tabs(['Règle Naive', 'MLP','Gridsearch','Randomsearch','XGBoost','Regression linéaire'])

    with naive:
        st.write("Pour la régle naive, le combattant :red[rouge] a été prédit comme gagnant dans tous les combats")
        plt.clf()
        cm=confusion_matrix(Y_test,Y_test_naive)
        confusion= sns.heatmap(cm, annot=True, fmt='g')
        # labels, title and ticks
        confusion.set_xlabel('Gagnant prédit');confusion.set_ylabel('Véritable gagnant'); 
        confusion.set_title('Matrice de confusion'); 
        confusion.xaxis.set_ticklabels(['Bleu', 'Rouge']); confusion.yaxis.set_ticklabels(['Bleu', 'Rouge'])
        st.pyplot(confusion.get_figure()) 
        st.markdown("Le taux de bonne classification est de **57.3%**")

    with mlp: #MLP de base
        plt.clf()
        cm=confusion_matrix(Y_test,df_MLP_base)
        confusion= sns.heatmap(cm, annot=True, fmt='g')
        # labels, title and ticks
        confusion.set_xlabel('Gagnant prédit');confusion.set_ylabel('Véritable gagnant'); 
        confusion.set_title('Matrice de confusion'); 
        confusion.xaxis.set_ticklabels(['Bleu', 'Rouge']); confusion.yaxis.set_ticklabels(['Bleu', 'Rouge'])
        st.pyplot(confusion.get_figure())
        st.write('''Toutes les variables de base du MLP de SKlearn ont étés utilisé à l'exception de nombre d'iterations et du "solver" puisque nous avons eu des problèmes de convergence.''') 
        st.write('Voici le code utlisé pour entainer le modèle')
        st.code("ufc = MLPClassifier(random_state=42,max_iter=4000,solver='sgd').fit(X_train_df, Y_train)")
        st.markdown("Le taux de bonne classification est de :red[**54.1%**]")

    with grid: #gridsearch
        plt.clf()
        cm=confusion_matrix(Y_test,df_xgboost)
        confusion= sns.heatmap(cm, annot=True, fmt='g')
        # labels, title and ticks
        confusion.set_xlabel('Gagnant prédit');confusion.set_ylabel('Véritable gagnant'); 
        confusion.set_title('Matrice de confusion'); 
        confusion.xaxis.set_ticklabels(['Bleu', 'Rouge']); confusion.yaxis.set_ticklabels(['Bleu', 'Rouge'])
        st.pyplot(confusion.get_figure()) 
        st.markdown("Le taux de bonne classification est de :red[**non-obtenu, trop long à rouler**]")
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
    with random: #randomsearch
        plt.clf()
        cm=confusion_matrix(Y_test,df_MLP_random)
        confusion= sns.heatmap(cm, annot=True, fmt='g')
        # labels, title and ticks
        confusion.set_xlabel('Gagnant prédit');confusion.set_ylabel('Véritable gagnant'); 
        confusion.set_title('Matrice de confusion'); 
        confusion.xaxis.set_ticklabels(['Bleu', 'Rouge']); confusion.yaxis.set_ticklabels(['Bleu', 'Rouge'])
        st.pyplot(confusion.get_figure()) 
        st.markdown("Le taux de bonne classification est de :green[**59,9%**]")

        st.write("Une fonction fut créée pour générer des chiffres entiers aléatoirement et une autre pour générer des chiffres à décimale")
        col1, col2= st.columns(2)

        with col1:
            st.write("Fonction générant des chiffres entiers")
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
            st.write("Fonction générant des chiffres décimaux")
            code_getthatfloat='''random.seed(12)
def getthatfloat(lower=0.0001,upper=1,count=30):
random.seed(12)
getthatlist=[]
for i in range(count):
x = round(random.uniform(lower,upper),4)
getthatlist.append(x)
return getthatlist'''
            st.code(code_getthatfloat,language='python')
        st.write("Des données furent générées et utilisées pour entrainer le MLP en choisissant 50 combinaisons différentes")
        code_randsearch='''ufc = MLPClassifier(random_state=42)
hidden_random=getthatint(10,300,50)
alpha_random=getthatfloat(0.0001,1,50)
learning_rate_init_random=getthatfloat(0.0001,1)
#number of iterations kept to 10, default setting for n_iter
param_random={'hidden_layer_sizes': hidden_random,
       'solver':['sgd','lbfgs'],
       'alpha':alpha_random,
       'batch_size':[256,512],
       'learning_rate_init':learning_rate_init_random,
       'momentum':[0.1,0.3,0.6,0.9],
       'max_iter':[1000,2000,3000,4000]}

# entrainement du modèle
clf_rand50=RandomizedSearchCV(ufc,param_random,n_iter=50,random_state=42)'''
        st.code(code_randsearch)
    with xg:#gridsearch
        plt.clf()
        cm=confusion_matrix(Y_test,df_xgboost)
        confusion= sns.heatmap(cm, annot=True, fmt='g')
        # labels, title and ticks
        confusion.set_xlabel('Gagnant prédit');confusion.set_ylabel('Véritable gagnant'); 
        confusion.set_title('Matrice de confusion'); 
        confusion.xaxis.set_ticklabels(['Bleu', 'Rouge']); confusion.yaxis.set_ticklabels(['Bleu', 'Rouge'])
        st.pyplot(confusion.get_figure()) 
        st.markdown("Le taux de bonne classification est de :green[**57.5%**]")

        st.write('Le code suivant fut ensuite utilisé pour créer le modèle')
        st.code('''from xgboost import XGBClassifier
ufcboost = XGBClassifier(random_state=42)
ufcboost.fit(X_train, Y_train)
test_pred = ufcboost.predict(X_test_df)''')

    with regression:
        plt.clf()
        cm=confusion_matrix(Y_test,df_logistique)
        confusion= sns.heatmap(cm, annot=True, fmt='g')
        # labels, title and ticks
        confusion.set_xlabel('Gagnant prédit');confusion.set_ylabel('Véritable gagnant'); 
        confusion.set_title('Matrice de confusion'); 
        confusion.xaxis.set_ticklabels(['Bleu', 'Rouge']); confusion.yaxis.set_ticklabels(['Bleu', 'Rouge'])
        st.pyplot(confusion.get_figure())
        st.markdown("Le taux de bonne classification est de :red[**57.1%**]")

        st.write('Le code suivant fut ensuite utilisé pour créer le modèle')
        st.code('''from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
pca=PCA(n_components=2)
x_train_pca=pca.fit_transform(X_train,Y_train)
x_test_pca=pca.transform(X_test)

#entrainement du modèle
logis_data=LogisticRegression(random_state=42).fit(x_train_pca, Y_train)''')

st.divider()

st.header("Analyse")
st.write("Nos résultats n'étant pas à la hauteur de nos attentes, une analyse nous a permis de comprendre les principales raisons de ce manque de performane.")
st.write("L'importance des varialbes de notre meilleur modèle (MLP utilisant Randomsearch) fut obtenue")

#création du graph
plt.clf()
barplot_eli5=sns.barplot(df_eli5.iloc[:,:10])
plt.xticks(rotation=30,ha='right')
plt.title("Poids de l'importance des variables")
plt.ylabel('Poids')
#st.pyplot(barplot_eli5.get_figure())
fig_html = mpld3.fig_to_html(barplot_eli5)
components.html(fig_html, height=600)

st.write("finalement, la représentation graphique du PCA a été crée pour remarquer que les gagnants sont difficilement différenciables, expliquant nos difficultées")
