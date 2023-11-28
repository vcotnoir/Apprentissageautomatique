import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score,confusion_matrix,ConfusionMatrixDisplay, precision_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

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

url_predictions_grid = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/clf_grid_pred.csv'
df_MLP_grid = pd.read_csv(url_predictions_grid)

url_preductions_XG = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/ufcboost_pred.csv'
df_xgboost = pd.read_csv(url_preductions_XG)

url_prediction_MLP = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/ufc_base_pred.csv'
df_MLP_base = pd.read_csv(url_prediction_MLP)

url_logistique = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/logis_predict.csv'
df_logistique = pd.read_csv(url_logistique)

url_eli5 = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/eli5_results.csv'
df_eli5 = pd.read_csv(url_eli5)

url_pca = 'https://raw.githubusercontent.com/vcotnoir/Apprentissageautomatique/main/df_pca.csv'
df_pca = pd.read_csv(url_pca)


shape_df=df.shape
shape_df_colonnes=shape_df[1]
shape_df_lignes=shape_df[0]

shape_df_processed=df_preprocessed.shape
shape_df_processed_colonnes=shape_df_processed[1]
shape_df_processed_lignes=shape_df_processed[0]

st.title('Apprentissage automatique - prédictions UFC')

intro_text='''But: Faire les meilleures prédictions possibles sur les combats UFC (problème de classification binaire)'''

st.header(intro_text)

st.header("Présentation des données")

st.write('''Des données issues des combats en UFC (Utimate Fighting Championship) ayant eu lieu entre 2010 et 2023 sont étés récoltées en utilisant Kaggle (2010-2022) et le "web-scrapping" (2023) afin de composer un jeu de données ayant pour but de faire de la classification binaire''')
st.write('''Pour fin de facilité, les données sont présentées, suivi des résultats. La méthodologie est présentée en dernier (analyse exploratoire et traitement des données).''')
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

st.divider()

st.header('Résultats')

explication = '''Les modèles suivants furent essayés, les résultats sont présentés initialement et le code par la suite.  
L'ordre de présentation représente l'ordre dans lequel les modèles furent testés.'''
st.markdown(explication)

with st.container():
    naive,mlp,grid,random,xg,regression = st.tabs(['Règle Naive', 'MLP','Gridsearch','Randomsearch','XGBoost','Regression linéaire'])

    with naive:
        st.write("Pour la règle naïve, le combattant :red[rouge] a été prédit comme gagnant dans tous les combats")
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
        st.markdown("Le taux de bonne classification est de :red[**54.1%**]")
        st.write('''Toutes les variables de base du MLP de SKlearn ont étés utilisé à l'exception de nombre d'iterations et du "solver" puisque nous avons eu des problèmes de convergence.''') 
        st.write('Voici le code utlisé pour entainer le modèle')
        st.code('''from sklearn.neural_network import MLPClassifier

#Entrainement du modèle
ufc = MLPClassifier(random_state=42,max_iter=4000,solver='sgd').fit(X_train_df, Y_train)''')
        

    with grid: #gridsearch
        plt.clf()
        cm=confusion_matrix(Y_test,df_MLP_grid)
        confusion= sns.heatmap(cm, annot=True, fmt='g')
        # labels, title and ticks
        confusion.set_xlabel('Gagnant prédit');confusion.set_ylabel('Véritable gagnant'); 
        confusion.set_title('Matrice de confusion'); 
        confusion.xaxis.set_ticklabels(['Bleu', 'Rouge']); confusion.yaxis.set_ticklabels(['Bleu', 'Rouge'])
        st.pyplot(confusion.get_figure()) 
        st.markdown("Le taux de bonne classification est de :green[**60.1%**]")
        # code utilisé
        code_grid='''from sklearn.model_selection import GridSearchCV

ufc = MLPClassifier(random_state=42)
random.seed(1234)

#Paramètres utilisés
param_grid={'hidden_layer_sizes': [10,20,75],
       'solver':['sgd','lbfgs'],
       'alpha':[0.0001,0.001],
       'batch_size':[256],
       'learning_rate_init':[0.001,0.01,0.1],
       'momentum':[0.1,0.3],
       'max_iter':[4000]}

#création du modèle
clf_grid=GridSearchCV(ufc,param_grid)

#Entrainement du modèle
clf_grid.fit(X_train,Y_train)'''
        
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

#Paramètres utilisés
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

    with xg:#xgboost
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

#création du modèle
ufcboost = XGBClassifier(random_state=42)
                
#entrainement du modèle
ufcboost.fit(X_train, Y_train)''')

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

#Création du PCA
pca=PCA(n_components=2)
x_train_pca=pca.fit_transform(X_train,Y_train)
x_test_pca=pca.transform(X_test)

#entrainement du modèle
logis_data=LogisticRegression(random_state=42).fit(x_train_pca, Y_train)''')

st.divider()

st.header("Analyse")
st.write("Nos résultats n'étant pas à la hauteur de nos attentes, une analyse nous a permis de comprendre les principales raisons de ce manque de performance.")
st.subheader("Permutation des données")
st.write("L'importance des variables de notre meilleur modèle (MLP utilisant Randomsearch) fut obtenue.")

#création du graph
plt.clf()
vert = sns.light_palette("seagreen",12,reverse=True)
barplot_eli5=sns.barplot(df_eli5.iloc[:,:10],palette=vert)
plt.xticks(rotation=30,ha='right')
plt.title("Poids de l'importance des variables")
plt.ylabel('Poids')
st.pyplot(barplot_eli5.get_figure())

analyse_graphique_importance = '''L'analyse des 10 variables les plus significatives nous permet de remarquer que les données plus avancées ont une utilité qui est limitée dans la prédiction.  
Des 10 variables, seulement 3 variables avancées se hissent dans la liste. 
- B_avg_SIG_STR_pct, qui représente le nombre de cours significatif sur le nombre de couts lancés, 
- B_avg_SIG_STR_landed qui représente le nombre de couts significatifs) 
- B_avg_TD_pct, qui représente le nombre d'amenées au sol réussies sur le nombre total essayé est la seule variable.  

4 des variables sont liées à l'âge ou aux mesures physiques des combattants
- B_age 
- R_age 
- height_dif 
- reach_dif  
  
Les 3 variables restantes sont en liens avec les fiches de victoires et défaites des combattants.'''
st.markdown(analyse_graphique_importance)
st.write("finalement, la représentation graphique du PCA a été créée pour remarquer que les gagnants sont difficilement différenciables, expliquant nos difficultés à obtenir de bonnes prédictions")

#création du PCA
st.subheader("Principal Component Analysis")
plt.clf()

df_pca_array=np.array(df_pca)
y=df_pca_array[:,2]
x=df_pca_array[:,[0,1]]

cdict={0:'Blue',1:'Red'}

for label in np.unique(y):
    plt.scatter(x[y==label, 0], x[y==label, 1], label=label, c=cdict[label])

plt.legend(["Bleu","Rouge"], title="Gagnant")
plt.title('PCA des données UFC')
plt.xlabel("Première composante principale")
plt.ylabel("deuxième composante principale")
st.pyplot(plt)

explication_PCA = '''Lorsque toutes les données sont projetées en 2 dimensions, la difficulté à obtenir des résultats devient claire.  
Toutes les données sont agglutinées d'une manière qui rend difficile de faire une différenciation entre les gagnants.  
Il est donc tout aussi difficile pour les modèles que nous avons utilisés de faire ce même genre de différenciation, expliquant ainsi les résultats obtenus.'''

st.markdown(explication_PCA)

st.divider()
st.header('Méthodologie')
st.subheader('Analyse exploratoire')
st.write('La corrélation entre les variables a été obtenue pour comprendre les données et savoir si certaines variables devraient être priorisées')


with st.container():
    correlation=sns.heatmap(df_correlation.corr(),cmap='coolwarm')
    correlation.set_title('Corrélation entre les variables')
    st.pyplot(correlation.get_figure())

st.write("Le paquet de visualisation Sweetviz a également été utilisé pour faire une analyse exploratoire des données")

st.subheader('''Traitement des données''')
Explanation_text_traitement_donnes='''  
1. Une sélection manuelle des variables a été fait suite à l'analyse des résultats de Sweetviz.  
2. Les données en lien avec les séries de victoires et de défaites furent combinées pour ne créer qu'une seule variable montrant la différence entre ces séries.  
3. Les données affairant aux victoires par décision et aux victoires par K.-O. furent regroupées. Par exemple, un combattant pouvait gagner une victoire par décision unanime pou partagé, ces données furent regroupées.  
4. Une variable montrant la différence de rang entre les combattants fut créée.
5. En raison de leur très petit nombre, les égalités furent retirées.  
6. Ajustement des anomalies ou des valeurs manquantes.
7. Les données furent standardisées et centrées avec une moyenne de 0 et un écart type de 1.
'''
st.markdown(Explanation_text_traitement_donnes)

st.divider()

