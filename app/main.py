import streamlit as st
import pandas as pd
import os
import pickle
import shap
import matplotlib.pyplot as plt
from utils import custom_load, get_prediction
from plots import plot_risk_score_gauge, plot_local_feature_importance, plot_global_feature_importance, plot_feature_analysis

st.set_page_config(page_title="Credit Scoring", layout="wide", page_icon='📊')

s_red = '#FF0051'
s_green = '#32CD32'

# @st.cache_data
def load_data(file_path, sep=','):
    return pd.read_csv(file_path, sep=sep, index_col=None)

@st.cache_data
def load_pickle(file_path):
    with open(file_path, 'rb') as model_file:
        return pickle.load(model_file)

@st.cache_data
def load_parquet(file_path):
    return pd.read_parquet(file_path)

        
def format_number(value):
    if pd.isna(value):
        return "-"
    return f"{int(value):,}".replace(",", " ")

@st.cache_data
def get_loan_informations(loan_id):
    client_infos = ['CODE_GENDER', 'DAYS_BIRTH', 'NAME_FAMILY_STATUS', 'OCCUPATION_TYPE', 'AMT_INCOME_TOTAL']
    client_df = app_test.loc[app_test['SK_ID_CURR'] == loan_id, client_infos]
    client_df.columns = ['Sexe', 'Age', 'Satut marital', 'Profession', 'Revenus annuels']
    
    client_df['Age'] = int(abs(client_df['Age'].iloc[0] / 365))
    client_df['Revenus annuels'] = client_df['Revenus annuels'].apply(format_number)
    client_df.fillna('-', inplace=True)
    client_df = client_df.astype(str).T
    client_df.columns = ['Demandeur']
    
    loan_infos = ['NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY']
    loan_df = app_test30.loc[app_test30['SK_ID_CURR'] == loan_id, loan_infos]
    loan_df.columns = ['Type', 'Montant', 'Annuités']
    loan_df['Montant'] = loan_df['Montant'].apply(format_number)
    loan_df['Annuités'] = loan_df['Annuités'].apply(format_number)
    loan_df = loan_df.astype(str).T
    loan_df.columns = ['Crédit']

    return client_df, loan_df

# Style des graphiques
plt.style.use('seaborn-v0_8-whitegrid')

# Répertoire courant
base_dir = os.path.dirname(os.path.abspath(__file__))

# Récupération des traductions des champs
fields_path = os.path.join(base_dir, '..', 'data', 'processed', 'descriptions.csv')
fields_df = load_data(fields_path, sep=";")

# Transformation du DataFrame en dictionnaire
field = {}
for _, row in fields_df.iterrows():
    champ = row['CHAMP']
    field[champ] = {
        'traduction': row['TRADUCTION'],
        'complement': row['COMPLEMENT']
    }
    
# Récupération du preprocessor
preprocessor_path = os.path.join(base_dir, '..', 'data', 'processed', 'preprocessor.pkl')
with open(preprocessor_path, 'rb') as f:
    preprocessor = custom_load(f)

# Chargement modèle
model_path = os.path.join(base_dir, '..', 'data', 'processed', 'model.pkl')
best_model = load_pickle(model_path)

# Chargement des données de test
app_test_path = os.path.join(base_dir, '..', 'data', 'raw', 'application_test.parquet')
app_test = load_parquet(app_test_path)

# Chargement du seuil de classification
threshold_path = os.path.join(base_dir, '..', 'data', 'processed', 'best_threshold.txt')
with open(threshold_path, 'r') as threshold_file:
    threshold = float(threshold_file.read())

# Récupération des top 30 features
top30_path = os.path.join(base_dir, '..', 'data', 'processed', 'top30_features.pkl')
top30_features = load_pickle(top30_path)
top30_features_sorted = sorted(top30_features)

# Chargement des données d'entrainement (30 top features)
app_train30_path = os.path.join(base_dir, '..', 'data', 'processed', 'application_train30.parquet')
app_train30 = load_parquet(app_train30_path)

# Chargement des données de test (30 top features)
app_test30_path = os.path.join(base_dir, '..', 'data', 'processed', 'application_test30.parquet')
app_test30 = load_parquet(app_test30_path)


# Url de l'API
api_url = 'https://failurescore-bc9f53f25e58.herokuapp.com/predict'
# api_url = 'http://127.0.0.1:5000/predict'

# Explainer SHAP
@st.cache_data
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(best_model)

# DataFrame explications top 30 features
explication = []
for feature in top30_features:
    exp = field[feature]['traduction']
    if not isinstance(field[feature]['complement'], float):
        exp = exp + f' ({field[feature]['complement']})'
    explication.append(exp)
explication_df = pd.DataFrame({'Caractéristique': top30_features, 'Explication': explication})

# Titre
st.markdown("<div style='text-align: center'> <h1>Credit Scoring - Prêt à Dépenser</h1> </div>", unsafe_allow_html=True)

# Bandeau latéral
with st.sidebar:
    # Sélection numéro demande de prêt
    loan_id = st.selectbox("Numéro de demande de crédit", app_test['SK_ID_CURR'].astype('int'))
    # Case à cocher Influence moyenne des caractéristiques
    show_global_importance = st.checkbox('Influences moyennes des caractéristiques sur les scores (historique des crédits accordés)')
    # Case à cocher Explication du score
    show_local_importance = st.checkbox(f'Influences des caractéristiques sur le score (demande n° {loan_id})')
    # Case à cocher Analyse de caractéristiques
    show_feature_analysis = st.checkbox(f'Analyse de 2 caractéristiques (demande n° {loan_id})')

# Filtre des données pour la demande sélectionnée
loan_data = app_test[app_test['SK_ID_CURR'] == loan_id]

# Remplacement des valeurs manquantes par None (JSON n'accepte pas les NaN)
loan_data_none = loan_data.map(lambda x: None if pd.isna(x) else x)

# Prédiction
if not loan_data.empty:
    # Appel de l'API pour prédiction
    prediction = get_prediction(loan_data_none, api_url)
    failure_probability = prediction['prediction_proba'][0]
    high_risk = prediction['prediction_class'][0]

    decision_color = s_red if high_risk else s_green
    
    st.markdown(f"""
        <div style='text-align: center'>
            <h4>Demande de crédit n° <span style='color: {decision_color};'>{loan_id}</span></h4>
        </div>
    """, unsafe_allow_html=True)
    
    left_margin, content, right_margin = st.columns([2, 5, 2])

    with content:
        st.divider()
        # Informations dossier et résultat
        col1, col2, col3 = st.columns([4, 1, 4])
            
        with col1:
            # Informations demandeur
            client, loan = get_loan_informations(loan_id)
            st.markdown(f"<div style='text-align: left'> <h5>Informations demandeur</h5> </div>", unsafe_allow_html=True)
            st.table(client)
            # Informations crédit
            st.markdown(f"<div style='text-align: left'> <h5>Informations crédit</h5> </div>", unsafe_allow_html=True)
            st.table(loan)
           
        with col3:
            # Jauge et indicateurs
            plot_risk_score_gauge(failure_probability, threshold, high_risk)
            # Calcul de la décision
            if high_risk:
                decision = 'REFUS'
                sign = '>'
            else:
                decision = 'ACCORD'
                sign = '<'
                
            # Affichage du score, seuil et décision
            html_code = f"""
            <div style="text-align: center;">
                <table style="width: 100%; border-collapse: collapse; border: none; margin: 0; padding: 0;">
                    <tr style="border: none; padding: 0;">
                        <th style="width: 30%; text-align: center; font-size: 14px; border: none; padding: 0;">Risque</th>
                        <th style="width: 5%; text-align: center; font-size: 14px; border: none; padding: 0;"></th>
                        <th style="width: 30%; text-align: center; font-size: 14px; border: none; padding: 0;">Seuil</th>
                        <th style="width: 5%; text-align: center; font-size: 14px; border: none; padding: 0;"></th>
                        <th style="width: 30%; text-align: center; font-size: 14px; border: none; padding: 0;">Décision</th>
                    </tr>
                    <tr style="border: none; padding: 0;">
                        <td style="text-align: center; font-size: 36px; border: none; padding: 0;">{failure_probability * 100:.2f} %</td>
                        <td style="text-align: center; font-size: 30px; border: none; padding: 0;">{sign}</td>
                        <td style="text-align: center; font-size: 36px; border: none; padding: 0;">{threshold * 100:.1f} %</td>
                        <td style="text-align: center; font-size: 30px; border: none; padding: 0;">=></td>
                        <td style="text-align: center; font-size: 36px; color: {decision_color}; border: none; padding: 0;">{decision}</td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(html_code, unsafe_allow_html=True)
     
        # Affichage de l'image d'importance globale
        if show_global_importance:
            plot_global_feature_importance(base_dir, top30_features, explication_df)
    
        # Graphique de feature importance locale
        if show_local_importance:
            plot_local_feature_importance(preprocessor, loan_data, explainer, loan_id, field)
    
        # Analyse de caractéristiques
        if show_feature_analysis:
            plot_feature_analysis(app_train30, app_test30, top30_features_sorted, loan_id, explication_df)

else:
    st.write("Sélectionner un numéro de demande valide.")