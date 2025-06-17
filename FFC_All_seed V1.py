# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 15:12:11 2025

@author: kaoutar
"""

import numpy as np
import pandas as pd
from scipy.integrate import simps
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt

# Base paths
base_path = r"C:\Users\kaoutar\Desktop\Projet 2\output\V3\VVcP-seed\CI1"
base_path2 = r"C:\Users\kaoutar\Desktop\Projet 2\output\V3\VVcP-seed\CI2"

# Datasets Real
data_Real1 = pd.read_csv(r"C:\Users\kaoutar\Desktop\Projet 2\output\V3\VVcP-Real\eig_CI1_Real.csv", delimiter=";")
data_Real2 = pd.read_csv(r"C:\Users\kaoutar\Desktop\Projet 2\output\V3\VVcP-Real\eig_CI2_Real.csv", delimiter=";")

dvr = data_Real1['dvr'].values

# --------------------------------------------------------------------------------------------------
# Fonctions de base
# --------------------------------------------------------------------------------------------------

def normalize_wavefunction(psi, r):
    norm = simps(psi**2, r)
    return psi / np.sqrt(norm)

def overlap_integral(psi_v, psi_v_prime, r):
    return simps(psi_v * psi_v_prime, r)

def FFC(psi_v, psi_v_prime, r):
    psi_v = normalize_wavefunction(psi_v, r)
    psi_v_prime = normalize_wavefunction(psi_v_prime, r)
    S = overlap_integral(psi_v, psi_v_prime, r)
    return S**2

def calculate_ffc(data1, data2, dvr):
    psi_v0 = data1['0'].values
    ffc_values = []
    for col in data2.columns[2:849]:
        psi_v_prime = data2[col].values
        ffc_value = FFC(psi_v0, psi_v_prime, dvr)
        ffc_values.append({'FFC': col, 'FFC Value': ffc_value})
    return ffc_values

def calculate_mse(group, real_values):
    merged = pd.merge(group, real_values, on='FFC', suffixes=('', '_real'))
    return mean_squared_error(merged['FFC Value'], merged['FFC Value_real'])

# --------------------------------------------------------------------------------------------------
# Chargement automatique
# --------------------------------------------------------------------------------------------------

def load_all_datasets_for_seed(seed, base_path, base_path2, iterations=8):
    models = {
        "APHYNITY": "AphNv",
        "Morse": "Morse",
        "Neural Network": "NN",
        "PhysiNet": "physinet",
        "Sequantial Phy-ML": "New"
    }

    all_data = {model_name: [] for model_name in models}
    for i in range(1, iterations + 1):
        for model_name, model_file in models.items():
            if model_name == "Morse":
                filename = f"{model_file}_dvr_results_iter{i}.csv"
            else:
                filename = f"{model_file}_dvr_results_iter{i}_seed{seed}.csv"

            filepath1 = os.path.join(base_path, filename)
            filepath2 = os.path.join(base_path2, filename)
            
            print(filepath1)
            print(filepath2)
            
            try:
                data1 = pd.read_csv(filepath1, delimiter=";")
                data2 = pd.read_csv(filepath2, delimiter=";")
                all_data[model_name].extend([data1, data2])
            except FileNotFoundError:
                print(f"Fichier manquant : {filepath1} ou {filepath2}")

    return all_data

# ---------------------------------------------------------------------------------------
# Calcul par seed
# ---------------------------------------------------------------------------------------

seeds = [2,4,10,22,23,28,29,30,32,42,52]
percentages = [5,10,15,20,25,30,35,40]

ffc_results_Real = calculate_ffc(data_Real1, data_Real2, dvr)
ffc_results_Real_df = pd.DataFrame(ffc_results_Real)

mse_results_by_seed = {}

for seed in seeds:
    data_files = load_all_datasets_for_seed(seed, base_path, base_path2)

    ffc_results = []
    for model_type, dataset_list in data_files.items():
        print(model_type)
        
        for idx in range(0, len(dataset_list), 2):
            data1 = dataset_list[idx]
            data2 = dataset_list[idx + 1]
            pourcentage = percentages[idx // 2]

            ffc = calculate_ffc(data1, data2, dvr)
            for entry in ffc:
                entry["Type"] = model_type
                entry["pourcentage"] = pourcentage
            ffc_results.extend(ffc)

    ffc_results_df = pd.DataFrame(ffc_results)

    mse_results = []
    for (model_type, p), group in ffc_results_df.groupby(['Type', 'pourcentage']):
        mse = calculate_mse(group, ffc_results_Real_df)
        mse_results.append({'Type': model_type, 'pourcentage': p, 'MSE': mse})

    mse_results_by_seed[seed] = pd.DataFrame(mse_results)

# Affichage exemple
for seed, df in mse_results_by_seed.items():
    print(f"\nMSE results for seed {seed}:")
    print(df)

# ---------------------------------------------------------------------------------------
# Concatener mse_results_by_seed
# ---------------------------------------------------------------------------------------

# Étape 1 : concaténer tous les résultats avec une colonne 'seed' pour garder la trace si besoin
all_mse_df = pd.concat(
    [df.assign(seed=seed) for seed, df in mse_results_by_seed.items()],
    ignore_index=True
)




#all_mse_df = pd.read_csv(r"C:\Users\kaoutar\Desktop\all_mse_grouped_raw.csv", delimiter=",")


# Grouper par Type et pourcentage et calculer la médiane Q1 et Q3 des MSE

stats_df = all_mse_df.groupby(['Type', 'pourcentage'])['MSE'].agg(
    Q1=lambda x: x.quantile(0.25),
    Median='median',
    Q3=lambda x: x.quantile(0.75)
).reset_index()


# ---------------------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------------------


plt.figure(figsize=(12, 6))

for model_type in stats_df['Type'].unique():
    model_data = stats_df[stats_df['Type'] == model_type]

    x = model_data['pourcentage'].to_numpy()
    y = model_data['Median'].to_numpy()
    q1 = model_data['Q1'].to_numpy()
    q3 = model_data['Q3'].to_numpy()

    plt.plot(x, y, label=model_type, marker='o')
    plt.fill_between(x, q1, q3, alpha=0.2)

plt.xlabel("Pourcentage de données")
plt.ylabel("MSE")
plt.yscale('log')
plt.title("MSE médian avec intervalle interquartile (Q1–Q3) par modèle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

























