# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:40:10 2024

@author: kaoutar
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# RMSE Mediane
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Initialisation d'un dictionnaire pour stocker les statistiques par ci_version
ci_stats = {}

ci_versions = ["CI2"] #["CI1", "CI2", "CI3", "CI4", "CI5", "CI6"]
percentages = ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%'] #, '60%', '80%']

for ci_version in ci_versions:
    # Chargement des données
    
    dataMorse = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3-original\{ci_version}\Morse_rmse_results.csv", delimiter=",")
    dataNew = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci_version}\New_rmse_results.csv", delimiter=",")
    #dataNew = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\testsNew\CI1\New8_rmse_results.csv", delimiter=",")
    
    dataAph = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci_version}\AphNv_rmse_results.csv", delimiter=",")  
    
    dataNN = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3-original\{ci_version}\NN_rmse_results.csv", delimiter=",")
    dataPhysinet = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\physinet\{ci_version}\physinet_rmse_results.csv", delimiter=",")
    
    # Définition des datasets et des colonnes
    datasets = {
        
        'Neural Network': dataNN,
        'Sequential Phy-ML': dataNew,
        'APHYNITY': dataAph,
        'PhysiNet': dataPhysinet
    }
    
    columns = ['iter_iter1', 'iter_iter2', 'iter_iter3','iter_iter4', 'iter_iter5', 
                   'iter_iter6', 'iter_iter7', 'iter_iter8'] #, 'iter_iter12', 'iter_iter16']
    
    #columns = ['iter_iter1', 'iter_iter2', 'iter_iter3','iter_iter4']
    
    colors = ['#000080','#800000','#ffa500','#32cd32']
    linestyles = ['-.','-.','-.','-.']  # Différents styles de ligne pour plus de distinction

    # Initialisation d'un dictionnaire pour stocker les statistiques par dataset pour la version courante
    all_stats = {}
    
    plt.figure(figsize=(15, 8))
    
    for (dataset_name, dataset), color, linestyle in zip(datasets.items(), colors, linestyles):
        # Calcul statistique
        box_stats_list = []
        for column in columns:
            iter_data = dataset[column]
            quartile_1 = iter_data.quantile(0.25)
            quartile_3 = iter_data.quantile(0.75)
            median = iter_data.median()
    
            box_stats = {
                'Column': column,
                'Q1 (25%)': quartile_1,
                'Q3 (75%)': quartile_3,
                'Median': median
            }
    
            box_stats_list.append(box_stats)
        
        # Enregistrement des statistiques pour chaque dataset dans le dictionnaire
        all_stats[dataset_name] = box_stats_list
    
        # Extraction des statistiques pour les tracés
        medians = [box_stats['Median'] for box_stats in box_stats_list]
        q1_values = [box_stats['Q1 (25%)'] for box_stats in box_stats_list]
        q3_values = [box_stats['Q3 (75%)'] for box_stats in box_stats_list]
    
        # Tracé des médianes et de la zone interquartile pour chaque dataset
        plt.plot(range(1, len(columns) + 1), medians, marker='o', linestyle=linestyle, linewidth=4, label=f'{dataset_name}', color=color)
    
        # Remplir la zone interquartile avec hachurage ou couleur
        if color == '#ffa500':
            plt.fill_between(range(1, len(columns) + 1), q1_values, q3_values, color='none', alpha=0.8, hatch='.', edgecolor='orange')
        else:
            plt.fill_between(range(1, len(columns) + 1), q1_values, q3_values, color=color, alpha=0.5)
    
    # Tracé des valeurs pour Morse avec une couleur distincte et un style différent
    plt.plot(range(1, len(columns) + 1), dataMorse.iloc[0].to_numpy(), marker='o', linestyle='-.', linewidth=4, label='Physical model', color='#ff0000')
    
    # Modification des étiquettes de l'axe des abscisses
    plt.xticks(ticks=range(1, len(columns) + 1), labels=percentages)
    
    # Ajout du titre, des étiquettes et des autres éléments de mise en forme
    # plt.title('A')
    plt.ylabel(r'$RMSE_{\mathrm{median}}$ (a.u.)')
    plt.xlabel('Percentage of data used for training')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    # Affichage du graphique
    plt.show()
    
    # Enregistrement des statistiques pour la version courante dans le dictionnaire global ci_stats
    ci_stats[ci_version] = all_stats

# Affichage ou utilisation de ci_stats si nécessaire
print(ci_stats)


'''
# RMSE Moyenne
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Initialisation d'un dictionnaire pour stocker les statistiques par ci_version
ci_stats = {}

ci_versions = ["CI2"] #["CI1", "CI2", "CI3", "CI4", "CI5", "CI6"]
percentages = ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%']

for ci_version in ci_versions:
    # Chargement des données
    dataMorse = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3-original\{ci_version}\Morse_rmse_results.csv", delimiter=",")
    dataNew8 = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\testsNew\{ci_version}\New8_rmse_results.csv", delimiter=",")
    dataNew = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3-original-consolide\{ci_version}\New2_rmse_results.csv", delimiter=",")
    
    
    dataAph = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci_version}\AphNv_rmse_results.csv", delimiter=",")  
    dataNN = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3-original\{ci_version}\NN_rmse_results.csv", delimiter=",")
    
    # Définition des datasets et des colonnes
    datasets = {
        #'dataAph': dataAph,
        #'dataNN': dataNN,
        #'dataPhyNet': dataPhyNet,
        'dataNew': dataNew,
        'dataNew8': dataNew8
    }
    
    columns = ['iter_iter1', 'iter_iter2', 'iter_iter3','iter_iter4', 'iter_iter5', 
                   'iter_iter6', 'iter_iter7', 'iter_iter8']
    
    colors = ['#000080','#ff0000','#000000','#32cd32'] # '#000000','#32cd32'  # Couleurs plus contrastées
    linestyles = ['-.','-.','-.','-.']  # Différents styles de ligne pour plus de distinction

    # Initialisation d'un dictionnaire pour stocker les statistiques par dataset pour la version courante
    all_stats = {}
    
    plt.figure(figsize=(10, 6))
    
    for (dataset_name, dataset), color, linestyle in zip(datasets.items(), colors, linestyles):
        # Calcul statistique
        box_stats_list = []
        for column in columns:
            iter_data = dataset[column]
            mean = iter_data.mean()  # Calcul de la moyenne
    
            box_stats = {
                'Column': column,
                'Mean': mean  # Utilisation de la moyenne uniquement
            }
    
            box_stats_list.append(box_stats)
        
        # Enregistrement des statistiques pour chaque dataset dans le dictionnaire
        all_stats[dataset_name] = box_stats_list
    
        # Extraction des statistiques pour les tracés
        means = [box_stats['Mean'] for box_stats in box_stats_list]
    
        # Tracé des moyennes pour chaque dataset
        plt.plot(range(1, len(columns) + 1), means, marker='o', linestyle=linestyle, linewidth=2, label=f'Moyennes - {dataset_name}', color=color)
    
    # Tracé des valeurs pour Morse avec une couleur distincte et un style différent
    plt.plot(range(1, len(columns) + 1), dataMorse.iloc[0], marker='o', linestyle='-.', linewidth=2, label='Morse', color='#6a5acd')
    
    # Modification des étiquettes de l'axe des abscisses
    plt.xticks(ticks=range(1, len(columns) + 1), labels=percentages)
    
    # Ajout du titre, des étiquettes et des autres éléments de mise en forme
    plt.title(f'Moyennes des RMSE - {ci_version}')
    plt.ylabel('RMSE')
    plt.xlabel('Tranches de données')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    # Affichage du graphique
    plt.show()
    
    # Enregistrement des statistiques pour la version courante dans le dictionnaire global ci_stats
    ci_stats[ci_version] = all_stats

# Affichage ou utilisation de ci_stats si nécessaire
print(ci_stats)


# Moyennes des RMSEs
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

# Dictionnaire pour stocker les moyennes des médianes pour chaque dataset et chaque colonne (iter_iter1, iter_iter2, etc.)
overall_median_means_all_columns = {
    'iter_iter1': {'dataAph1': [], 'dataNN': [], 'dataNew': [], 'dataPhyNet': []},
    'iter_iter2': {'dataAph1': [], 'dataNN': [], 'dataNew': [], 'dataPhyNet': []},
    'iter_iter3': {'dataAph1': [], 'dataNN': [], 'dataNew': [], 'dataPhyNet': []},
    #'iter_iter4': {'dataAph1': [], 'dataNN': [], 'dataNew': [], 'dataPhyNet': []}
}

# Parcours de chaque version de CI
for ci_version, datasets in ci_stats.items():
    for dataset_name, stats in datasets.items():
        for column in overall_median_means_all_columns.keys():
            # Filtrer les stats pour la colonne spécifique et extraire la médiane
            medians = [stat['Median'] for stat in stats if stat['Column'] == column]
            if medians:
                overall_median_means_all_columns[column][dataset_name].extend(medians)

# Calcul des moyennes des médianes pour chaque dataset et chaque colonne sur toutes les CI versions
average_median_per_dataset_all_columns = {
    column: {dataset: np.mean(medians) for dataset, medians in datasets.items()}
    for column, datasets in overall_median_means_all_columns.items()
}

# Conversion du dictionnaire en DataFrame pour visualisation
average_median_all_columns_df = pd.DataFrame(average_median_per_dataset_all_columns)

# Affichage du DataFrame
print(average_median_all_columns_df)


# Plot Moyennes
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

# Définition des abscisses
x_values = [5, 10, 15, 20]

# Création de la figure
plt.figure(figsize=(10, 6))

# Tracé des valeurs pour chaque dataset
for dataset_name, values in average_median_all_columns_df.iterrows():
    plt.plot(x_values, values, marker='o', linestyle='-', linewidth=2, label=dataset_name)

# Ajout des titres et des labels
plt.title("Moyenne des Médianes par Dataset et Colonne")
plt.xlabel("Abscisses (5, 10, 15, 20)")
plt.ylabel("Valeurs des Médianes")
plt.legend()
#plt.yscale('log')
plt.grid(True)

# Affichage du graphique
plt.show()


# Plot supp
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt


Aph4 = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3\CI2\AphNv_rmse_results.csv", delimiter=",")
Aph5 = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3\CI2\AphNv_rmse_results.csv", delimiter=",")
Aph6 = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3\CI2\AphNv_rmse_results.csv", delimiter=",")
Aph7 = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3\CI2\AphNv_rmse_results.csv", delimiter=",")
Aph8 = pd.read_csv(fr"C:\Users\kaoutar\Desktop\Projet 2\output\V3\CI2\AphNv_rmse_results.csv", delimiter=",")


def plot_boxplots(dataframe, column_names):
    
    # Vérifier si les colonnes existent et convertir en numérique
    for column_name in column_names:
        if column_name not in dataframe.columns:
            raise ValueError(f"La colonne '{column_name}' n'existe pas dans le dataframe.")
        # Convertir les données en numériques, en gérant les valeurs non convertibles
        dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce')

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [dataframe[col].dropna() for col in column_names],  # Exclure les valeurs NaN
        vert=True,
        patch_artist=True,
        labels=column_names
    )
    plt.title("Boîtes à moustaches RMSE")
    plt.ylabel("Valeurs")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Exemple d'utilisation
data = {
    "Valeurs4": Aph4['iter_iter4'],
    "Valeurs5": Aph5['iter_iter5'],
    "Valeurs6": Aph5['iter_iter6'],
    "Valeurs7": Aph5['iter_iter7'],
    "Valeurs8": Aph5['iter_iter8']
    
}

df = pd.DataFrame(data)

# Appeler la fonction pour tracer les boîtes à moustaches
plot_boxplots(df, ["Valeurs4", "Valeurs5", "Valeurs6", "Valeurs7", "Valeurs8"])

'''


