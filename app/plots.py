import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamviz import gauge
import plotly.graph_objects as go
import shap

s_red = '#FF0051'
s_green = '#32CD32'
s_blue = "#1E90FF"

@st.cache_data
def plot_risk_score_gauge(failure_probability, classification_threshold, high_risk):
    threshold = classification_threshold * 100
    probability = failure_probability * 100
    color_bar = 'limegreen' if probability < threshold else s_red
    # Jauge de risque
    layout = go.Layout(
        width=600,  # Width in pixels
        height=350,  # Height in pixels
        margin={'b': 0, 't': 30}
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Indicator(
        mode='gauge+number+delta',
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Score de risque',
               'font': {'size': 24}},
        delta={'reference': threshold,
               'increasing': {'color': s_red},
               'decreasing': {'color': 'limegreen'}},
        gauge={'axis': {'range': [None, 100],
                        'tickmode': 'array',
                        'tickvals': [0, 20, 40, threshold, 60, 80, 100],
                        'ticktext': ['0', '20', '40', str(round(threshold, 1)), '60', '80', '100'],
                        'tickfont': {'size': 16, 'weight': 'bold', 'color': 'gray'}},
               'bar': {'color': color_bar, 'thickness': 0.7},
               'borderwidth': 2,
               'bordercolor': "gray",
               'threshold': {'line': {'color': s_blue, 'width': 3},
                            'thickness': 1,
                            'value': threshold}}
    ))

    st.plotly_chart(fig)


def plot_global_feature_importance(base_dir, top30_features, explication_df):
    st.divider()
    st.markdown(f"<div style='text-align: center'> <h3>Influence des caractéristiques (historique des crédits)</h3> </div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 4.2])

    with col1:
        st.image(os.path.join(base_dir, 'assets', 'images', 'global_importance_top20.svg'), use_column_width=True)

    with col2:
        st.markdown(f"<div style='text-align: center'> <h5>Signification des caractéristiques</h5> </div>", unsafe_allow_html=True)
        st.dataframe(explication_df.iloc[:20], hide_index=True, height=737)


@st.cache_data
def generate_local_feature_importance(_preprocessor, loan_data, _explainer, loan_id):
    processed_data = _preprocessor.transform(loan_data.drop(columns=['SK_ID_CURR']))
    shap_values = _explainer(processed_data)

    fig, ax = plt.subplots(figsize=(10, 3))

    shap.plots.bar(shap_values[0], max_display=16, show=False)
    bars = ax.patches
    annotations = ax.texts

    # Modification couleur : bleu -> vert pour les valeurs SHAP négatives
    # (vert = diminution du risque)
    for i, bar in enumerate(bars):
        if bar.get_width() < 0:  # Si la valeur SHAP est négative
            bar.set_color('limegreen')
            annotations[i].set_color('limegreen')
    
    # Modification du texte "Sum of X other features"
    ticks = ax.get_yticklabels()
    features = []
    for tick in ticks:
        if tick.get_text().startswith("Sum of"):
            tick.set_text("Autres caractéristiques")
        else:
            features.append(tick.get_text())
    ax.set_yticklabels(ticks, fontsize=10)
    
    ax.set_title("Influence des caractéristiques de la demande de prêt\nsur la probabilité de défaut de remboursement", fontsize=14, pad=15)
    ax.set_xlabel("Contribution sur le risque de défaut", fontsize=12)
    ax.set_ylabel("Caractéristiques de la demande de prêt", fontsize=12)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    
    return fig, features


def get_feature_name(feature):
    feature = feature.split()[-1]
    while feature.upper() != feature:
        feature = '_'.join(feature.split('_')[:-1])
    if feature.split('_')[-1] in ('M', 'F', 'MONDAY', 'TUESDAY', 'THURSDAY', 'FRIDAY', 'WEDNESDAY', 'SATURDAY', 'SUNDAY', 'XNA', 'ANOM'):
        feature = '_'.join(feature.split('_')[:-1])
    return feature


def plot_local_feature_importance(preprocessor, loan_data, explainer, loan_id, field):
    st.divider()
    st.markdown(f"<div style='text-align: center'> <h3>Explication du score de risque (demande n° {loan_id})</h3> </div>", unsafe_allow_html=True)

    col1, col2 = st.columns([7, 4])

    with col1:
        fig, features = generate_local_feature_importance(preprocessor, loan_data, explainer, loan_id)
        st.pyplot(fig)

    with col2:
        st.markdown(f"<div style='text-align: center'> <h5>Signification des caractéristiques</h5> </div>", unsafe_allow_html=True)
        # st.write(features)
        feat = []
        explication = []
        for feature in features[:15]:
            feature = get_feature_name(feature)
            if feature in field:
                exp = field[feature]['traduction']
                if not isinstance(field[feature]['complement'], float):
                    exp = exp + f' ({field[feature]['complement']})'
                explication.append(exp)
                feat.append(feature)
        # explication = [field[feature]['traduction'] for feature in top30_features]
        explication_df = pd.DataFrame({'Caractéristique': feat, 'Explication': explication})
        height = (len(explication_df) + 1) * 35 + 2
        st.dataframe(explication_df, hide_index=True, height=height)


def format_number(value):
    """Formate les nombres en fonction de leur grandeur."""
    if pd.isna(value):
        return "non disponible"
    elif value < 1:
        return f"{value:.3f}"
    elif value < 1000:
        return f"{int(value)}"
    elif value < 1_000_000:
        return f"{value / 1_000:.1f}K".replace(".0", "")
    else:
        return f"{value / 1_000_000:.1f}M".replace(".0", "")

@st.cache_data
def generate_categorical_feature_plot(app_train30, app_test30, feature, loan_id):
    current_value = app_test30.loc[app_test30['SK_ID_CURR'] == loan_id, feature].values[0]
    if pd.isna(current_value):
        current_value_label = "non disponible"
    elif type(current_value) != float or type(current_value) != int:
        current_value_label = current_value
    else:
        current_value_label = format_number(current_value)

    num_modalities = app_train30[feature].nunique()
    fig_height = 2 + num_modalities * 0.25
    
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    # Compte des modalités
    data = app_train30[[feature, 'TARGET']]
    counts = data.groupby([feature, 'TARGET']).size().unstack()

    # Normalisation des proportions
    proportions = counts.div(counts.sum(axis=0), axis=1)
    
    # Tri des proportions pour inverser l'ordre d'affichage
    proportions = proportions.sort_index(ascending=False)

    # Graphique en barres côte-à-côte
    proportions.plot(kind='barh', stacked=False, color=["limegreen", s_red], ax=ax)
    
    # Ajout de la ligne pour la current_value
    if not pd.isna(current_value):
        current_index = proportions.index.get_loc(current_value)
        ax.axhline(y=current_index, color=s_blue, linestyle='--', label=f"Demande n° {loan_id} : {current_value_label}")

    ax.set_xlabel('Densité')
    ax.set_ylabel(feature)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.set_title(f'Densité de probabilité des crédits remboursés / en défaut\npar {feature}', fontsize=14, pad=70)   
    ax.legend([f"Demande n° {loan_id} : {current_value_label}", "Crédits remboursés", "Crédits en défaut"], loc=3, bbox_to_anchor=(0., 1.02, 1., .102), ncol=1)

    fig.subplots_adjust(top=0.85, bottom=0.2)
    return fig
    

@st.cache_data
def generate_numeric_feature_plot(app_train30, app_test30, feature, loan_id):
    current_value = app_test30.loc[app_test30['SK_ID_CURR'] == loan_id, feature].values[0]
    if pd.isna(current_value):
        current_value_label = "non disponible"
    elif type(current_value) != float or type(current_value) != int:
        current_value_label = current_value
    else:
        current_value_label = format_number(current_value)
    
    fig, ax = plt.subplots(figsize=(8, 3))

    # Ajuster les données pour l'échelle
    scale = 1
    if app_train30[feature].max() >= 1_000_000:
        scale = 1_000_000
        ax.set_xlabel(f"{feature} (millions)")
    elif app_train30[feature].max() >= 1_000:
        scale = 1_000
        ax.set_xlabel(f"{feature} (milliers)")
    else:
        ax.set_xlabel(feature)
    
    # KDE plot des crédits remboursés dans les temps
    sns.kdeplot(app_train30.loc[app_train30['TARGET'] == 0, feature] / scale, label='Crédits remboursés', color="limegreen", ax=ax)
    
    # KDE plot des crédits non rembousés dans les temps
    sns.kdeplot(app_train30.loc[app_train30['TARGET'] == 1, feature] / scale, label='Crédits en défaut', color=s_red, ax=ax)
    
    # Ligne pointillée pour la valeur actuelle
    if not pd.isna(current_value):
        ax.axvline(x=current_value / scale, color=s_blue, linestyle='--', label=f"Demande n° {loan_id} : {current_value_label}")
    
    # Légendes
    ax.set_ylabel('Densité')
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.set_title(f'Densité de probabilité des crédits remboursés / en défaut\npar {feature}', fontsize=14, pad=10)
    ax.legend()
    return fig



def plot_numeric_feature(app_train30, app_test30, feature, loan_id):
    fig = generate_numeric_feature_plot(app_train30, app_test30, feature, loan_id)
    st.pyplot(fig)


def plot_categorical_feature(app_train30, app_test30, feature, loan_id):
    fig = generate_categorical_feature_plot(app_train30, app_test30, feature, loan_id)
    st.pyplot(fig)
    

def plot_feature(app_train, app_test, feature, loan_id):
    # Vérifier si la feature est catégorielle avec <= 24 modalités
    if app_train[feature].nunique() <= 24 and (app_train[feature].dtype == np.int64 or app_train[feature].dtype == object):
        plot_categorical_feature(app_train, app_test, feature, loan_id)
    else:
        plot_numeric_feature(app_train, app_test, feature, loan_id)
    return


@st.cache_data
def generate_bivariate_barplot(app_train, feature1, feature2, app_test, loan_id):
    feature1_current_value = app_test.loc[app_test['SK_ID_CURR'] == loan_id, feature1].values[0]
    feature2_current_value = app_test.loc[app_test['SK_ID_CURR'] == loan_id, feature2].values[0]

    if pd.isna(feature1_current_value) or pd.isna(feature2_current_value):    
        return None

    # Discrétiser les variables numériques si nécessaire
    if np.issubdtype(app_train[feature1].dtype, np.number):
        bins1 = pd.qcut(app_train[feature1], 5, duplicates='drop').cat.categories
        app_train[feature1] = pd.cut(app_train[feature1], bins=bins1)
        app_test[feature1] = pd.cut(app_test[feature1], bins=bins1)

    if np.issubdtype(app_train[feature2].dtype, np.number):
        bins2 = pd.qcut(app_train[feature2], 5, duplicates='drop').cat.categories
        app_train[feature2] = pd.cut(app_train[feature2], bins=bins2)
        app_test[feature2] = pd.cut(app_test[feature2], bins=bins2)

    # Tableau croisé dynamique pour compter les occurrences avec normalisation des proportions
    cross_tab = pd.crosstab([app_train[feature1], app_train[feature2]], app_train['TARGET'])
    cross_tab = cross_tab.div(cross_tab.sum(axis=0), axis=1)

    # Tri pour inverser l'ordre d'affichage
    cross_tab = cross_tab.sort_index(ascending=False)

    # Ajuster la hauteur du graphique en fonction du nombre de barres
    num_bars = len(cross_tab)
    height = num_bars * 0.4 + 4  # 0.3 pour chaque barre et 4 pour la base

    fig, ax = plt.subplots(figsize=(10, height))

    # Barres empilées
    cross_tab.plot(kind='barh', stacked=False, color=["limegreen", s_red], ax=ax)

    # Mise en évidence des valeurs actuelles
    feature1_value = app_test.loc[app_test['SK_ID_CURR'] == loan_id, feature1].values[0]
    feature2_value = app_test.loc[app_test['SK_ID_CURR'] == loan_id, feature2].values[0]
    current_index = cross_tab.index.get_loc((feature1_value, feature2_value))
    
    ax.axhline(y=current_index, color=s_blue, linestyle='--', label=f'Demande n° {loan_id}')

    # Configuration des labels sur deux lignes
    ax.set_yticklabels([f"{idx[0]}\n{idx[1]}" for idx in cross_tab.index])

    ax.set_xlabel('Densité')
    ax.set_ylabel(f"{feature1}\n{feature2}")
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.set_title(f'Densité de probabilité des crédits remboursés / en défaut\npar {feature1} et {feature2}', fontsize=14, pad=70)
    ax.legend([f"Demande n° {loan_id}", "Crédits remboursés", "Crédits en défaut"], loc=3, bbox_to_anchor=(0., 1.02, 1., .102), ncol=1)
    fig.subplots_adjust(top=0.85, bottom=0.2)
    return fig


def plot_bivariate_barplot(app_train, feature1, feature2, app_test, loan_id):
    fig = generate_bivariate_barplot(app_train, feature1, feature2, app_test, loan_id)
    if fig:
        st.pyplot(fig)


# @st.cache_data
def generate_bivariate_scatterplot(app_train, feature1, feature2, app_test, loan_id):
    feature1_current_value = app_test.loc[app_test['SK_ID_CURR'] == loan_id, feature1].values[0]
    feature2_current_value = app_test.loc[app_test['SK_ID_CURR'] == loan_id, feature2].values[0]
    if pd.isna(feature1_current_value) or pd.isna(feature2_current_value):
        message = f"""
        Les graphiques d'analyse croisée ne sont pas disponibles, car au moins une caractéristique n'est pas renseignée :  
        {feature1} : {feature1_current_value}  
        {feature2} : {feature2_current_value}
        """
        st.warning(message, icon="⚠️")
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
    
        # Palette de couleurs personnalisée
        palette = {0: "limegreen", 1: s_red}
        
        # Scatter plot avec une seule commande
        sns.scatterplot(data=app_train, x=feature1, y=feature2, hue=app_train['TARGET'], s=5, palette=palette, alpha=0.6, ax=ax)
        
        # Mise en évidence current_values
        feature1_value = app_test.loc[app_test['SK_ID_CURR'] == loan_id, feature1].values[0]
        feature2_value = app_test.loc[app_test['SK_ID_CURR'] == loan_id, feature2].values[0]
        ax.scatter(feature1_value, feature2_value, color=s_blue, edgecolor='black', s=30, label=f'Demande n° {loan_id}', alpha=0.6)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        
        ax.set_title(f'Répartition des crédits remboursés / en défaut\npar {feature1} et {feature2}', fontsize=14, pad=10)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['Crédits remboursés', 'Crédits en défauts', f'Demande n° {loan_id}'])
    except:
        return None
    return fig


def plot_bivariate_scatterplot(app_train, feature1, feature2, app_test, loan_id):
    fig = generate_bivariate_scatterplot(app_train, feature1, feature2, app_test, loan_id)
    if fig:
        st.pyplot(fig)
        return True
    else:
        return False
    
def plot_feature_analysis(app_train30, app_test30, top30_features, loan_id, explication_df):
    st.divider()
    st.markdown(f"<div style='text-align: center'> <h3>Analyse de deux caractéristiques</h3> </div>", unsafe_allow_html=True)

    # Création de la liste des choix avec explications
    choices = explication_df['Caractéristique'] + ' : ' + explication_df['Explication']
    choices = choices.sort_values()

    # Sélection des valeurs par défaut
    default_choices = [choices.iloc[0], choices.iloc[19]]  # AMT_CREDIT et NAME_
    
    with st.form('bivariate_analysis'):
        features = st.multiselect(
            "Sélectionner 2 caractéristiques à analyser",
            choices,
            default=default_choices,  # Définit les éléments par défaut
            max_selections=2,
            placeholder='Choisir une caractéristique'
        )

        # Affichage des deux cases à cocher côte à côte
        col1, col2, col3 = st.columns(3)

        with col1:
            univariate_graphs = st.checkbox(f'Graphiques d\'analyse des caractéristiques')

        with col2:
            bivariate_graphs = st.checkbox(f'Graphiques d\'analyse croisée')

        with col3:
            submit = st.form_submit_button('Afficher')
    

    if submit:
        if len(features) != 2:
            warning_box = st.warning("Veuillez sélectionner 2 caractéristiques.")
        else:
            feature1, feature2 = [feature.split(' : ')[0] for feature in features]
            st.write(features[0])
            st.write(features[1])

            c1, c2, c3 = st.columns([1, 6, 1])
            with c2:

                # Affichage des graphiques de densité univariés
                if univariate_graphs:
                    plot_feature(app_train30, app_test30, feature1, loan_id)
                    plot_feature(app_train30, app_test30, feature2, loan_id)

                # Analyse graphique bivariée
                if bivariate_graphs:
                    plot_bivariate_barplot(app_train30, feature1, feature2, app_test30, loan_id)
                    result = plot_bivariate_scatterplot(app_train30, feature1, feature2, app_test30, loan_id)
                    if result == False:
                        result = plot_bivariate_scatterplot(app_train30, feature1, feature2, app_test30, loan_id)

# def plot_feature_analysis(app_train30, app_test30, top30_features, loan_id, explication_df):
#     st.divider()
#     st.markdown(f"<div style='text-align: center'> <h3>Analyse de deux caractéristiques</h3> </div>", unsafe_allow_html=True)

#     choices = explication_df['Caractéristique'] + ' : ' + explication_df['Explication']
#     choices = choices.sort_values()
    
#     with st.form('bivariate_analysis'):

#         features = st.multiselect(
#             "Sélectionner 2 caractéristiques à analyser",
#             choices,
#             # ['AMT_CREDIT', 'NAME_CONTRACT_TYPE'],
#             max_selections=2,
#             placeholder='Choisir une caractéristique'
#         )
        
#         # Affichage des deux cases à cocher côte à côte
#         col1, col2, col3 = st.columns(3)
    
#         with col1:
#             univariate_graphs = st.checkbox(f'Graphiques d\'analyse des caractéristiques')
   
#         with col2:
#             bivariate_graphs = st.checkbox(f'Graphiques d\'analyse croisée')
            
#         with col3:
#             submit = st.form_submit_button('Afficher')

#     if submit:
#         if len(features) != 2:
#             warning_box = st.warning("Veuillez sélectionner 2 caractéristiques.")
#         else:
#             c1, c2, c3 = st.columns([1, 6, 1])
#             feature1, feature2 = [feature.split(' : ')[0] for feature in features]

#             with c2:
#                 st.write(features[0])
#                 st.write(features[1])
#                 # Affichage des graphiques de densité univariés
#                 if univariate_graphs:
#                     plot_feature(app_train30, app_test30, feature1, loan_id)
#                     plot_feature(app_train30, app_test30, feature2, loan_id)
            
#                 # Analyse graphique bivariée
#                 if bivariate_graphs:
#                     plot_bivariate_barplot(app_train30, feature1, feature2, app_test30, loan_id)
#                     result = plot_bivariate_scatterplot(app_train30, feature1, feature2, app_test30, loan_id)
#                     if result == False:
#                         result = plot_bivariate_scatterplot(app_train30, feature1, feature2, app_test30, loan_id)


