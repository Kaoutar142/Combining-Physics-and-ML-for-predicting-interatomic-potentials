# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:31:37 2024

@author: kaoutar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
                                                                                                                                                                    
# Datas
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# Charger les données
data0 = pd.read_csv(r"C:\Users\kaoutar\Desktop\pec_sigmag.txt", header=None, delimiter=" ")
data0.rename(columns={0:'Distance',1:'CI1',2:'CI2',3:'CI3',4:'CI4',5:'CI5',6:'CI6',7:'CI7',8:'CI8'}, inplace=True)
data2 = data0[data0["Distance"] < 20]

# Fonction pour sélectionner des pourcentages croissants de données et les stocker dans un dictionnaire
def select_data_in_steps(data, percentage, iterations):
    selected_data_dict = {}  # Dictionnaire pour stocker les datasets sélectionnés
    selected_data_test_dict = {}
    selected_data = pd.DataFrame()  # DataFrame temporaire pour accumuler les points sélectionnés
    all_selected_indices = set()  # Ensemble pour suivre les indices déjà sélectionnés

    for i in range(1, iterations + 1):
        # Calculer le nombre total de points requis pour l'itération actuelle
        total_points_to_select = int(len(data) * percentage / 100) * i
        
        # Calculer le nombre de points supplémentaires nécessaires
        additional_points_to_select = total_points_to_select - len(all_selected_indices)
        #print(f"Itération {i}: Total points = {total_points_to_select}, Additional points = {additional_points_to_select}")
        
        if additional_points_to_select > 0:
            # Générer des indices condensés au début et plus espacés vers la fin
            remaining_indices = sorted(set(range(len(data))) - all_selected_indices)

            # Utiliser une distribution basée sur la racine carrée pour rendre la sélection condensée au début
            sqrt_indices = np.linspace(0, np.sqrt(len(remaining_indices) - 1), additional_points_to_select)
            new_indices = np.unique((sqrt_indices ** 2).astype(int))  # Remettre à l'échelle avec **2
            
            # Ajouter seulement les indices restants
            selected_new_indices = [remaining_indices[i] for i in new_indices if i < len(remaining_indices)]
            all_selected_indices.update(selected_new_indices)
            
            # Sélectionner les nouvelles données en utilisant les nouveaux indices
            new_data = data.iloc[selected_new_indices].drop_duplicates().sort_index()
            
            # Ajouter les nouvelles données au dataset sélectionné
            selected_data = pd.concat([selected_data, new_data]).drop_duplicates().sort_index()
            
            '''
            # Plots datas selects
            plt.plot(data2['Distance'], data2['CI2'], 'b-', label='data', linewidth=1)
            plt.scatter(selected_data['Distance'], selected_data['CI2'], label='Test values',c='red', s=4)
            plt.show()
            '''
            
        # Sélectionner le reste des points pour le test
        selected_data_test = data[~data.index.isin(selected_data.index)]

        # Stocker le dataset correspondant à l'itération actuelle dans le dictionnaire
        selected_data_dict[f"iter{i}"] = selected_data.copy()
        selected_data_test_dict[f"iter{i}"] = selected_data_test.copy()
        
    return selected_data_dict, selected_data_test_dict


# Sélection des données en étapes et stockage dans un dictionnaire
data_selected_dict, selected_data_test_dict = select_data_in_steps(data2, 5, 8)


# Descente de Gradient
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# Initialiser les dictionnaires pour stocker les résultats
all_rmse_results = {}
all_mse_results = {}

def morse_torch(x, De, a, re, v):
    return (De * ((1 - torch.exp(-a * (x - re)))**2)) + v

# Modèle PyTorch pour ajuster les paramètres
class MorseModel(torch.nn.Module):
    def __init__(self):
        super(MorseModel, self).__init__()
        self.De = torch.nn.Parameter(torch.tensor(4, dtype=torch.float32))
        self.a = torch.nn.Parameter(torch.tensor(3, dtype=torch.float32))
        self.re = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
    def forward(self, x):
        return morse_torch(x, self.De, self.a, self.re, self.v)

# Boucle pour chaque CI (CI1 à CI6)
for ci in ['CI1'] : #, 'CI2', 'CI3', 'CI4', 'CI5', 'CI6']:

    # Initialiser les dictionnaires pour stocker les résultats pour chaque CI
    all_rmse_results = {}
    all_mse_results = {}

    all_test_results = {}
    all_dvr_results = {}
    
    keys_to_use = list(data_selected_dict.keys())[0:8]
    
    for i in keys_to_use:
        
    # Boucle pour chaque pourcentage dans `data_selected_dict`
    #for i in data_selected_dict:

        # Data Train
        x_data = data_selected_dict[i]['Distance']
        y_data = data_selected_dict[i][ci]
        
        x_data = x_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)
        
        # Data Test
        x_data_test = selected_data_test_dict[i]['Distance']
        y_data_test = selected_data_test_dict[i][ci]
        
        x_data_test = x_data_test.reset_index(drop=True)
        y_data_test = y_data_test.reset_index(drop=True)
        
        # Convertir en tenseurs 2D PyTorch
        X_train = torch.tensor(x_data, dtype=torch.float32).reshape(-1, 1)
        y_train = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(x_data_test, dtype=torch.float32).reshape(-1, 1)
        y_test = torch.tensor(y_data_test, dtype=torch.float32).reshape(-1, 1)
        
        # Instancier le modèle
        model = MorseModel()
        
        # Optimiseur
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Fonction de coût (Mean Squared Error)
        criterion = torch.nn.MSELoss()
        
        dvr00 = np.arange(0.5,9,0.01)
        dvr0 = torch.tensor(dvr00, dtype=torch.float32)
        dvr = dvr0.unsqueeze(1)
        
        # Entraînement
        for epoch in range(100000):  # Réduire le nombre d'époques pour tester
            # Prédiction du modèle
            y_pred = model(X_train)
        
            # Calcul de la perte
            loss = criterion(y_pred, y_train)
            #if epoch % 100 == 0:
                #print(f"Epoch {epoch} - Loss: {loss.item()} - CI: {ci}")
            
            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Rétropropagation
            loss.backward()
        
            # Mise à jour des paramètres
            optimizer.step()
        
        # Paramètres ajustés
        De_fit = model.De.item()
        a_fit = model.a.item()
        re_fit = model.re.item()
        v_fit = model.v.item()

        #print(f"Paramètres ajustés (De, a, re, v) pour {ci}: {De_fit}, {a_fit}, {re_fit}, {v_fit}")

        # Prédictions et métriques d'évaluation
        y_pred_train = morse_torch(X_train, De_fit, a_fit, re_fit, v_fit)
        y_pred_test = morse_torch(X_test, De_fit, a_fit, re_fit, v_fit)
        
        y_pred_dvr = model(dvr)
        y_pred_dvr = pd.DataFrame(y_pred_dvr.detach().numpy())
        
        mse_Tr = ((y_pred_train - y_train)**2).mean()
        rmse_Tr = torch.sqrt(mse_Tr)
        
        mse_Ts = ((y_pred_test - y_test)**2).mean()
        rmse_Ts = torch.sqrt(mse_Ts)

        # Plots
        '''
        plt.text(0.3, 0.7, 'MSE Test: {:.8f}'.format(mse_Ts), transform=plt.gca().transAxes, fontsize=10, color='blue')
        plt.text(0.3, 0.65, 'RMSE Test: {:.8f}'.format(rmse_Ts), transform=plt.gca().transAxes, fontsize=10, color='blue')
        plt.plot(data2['Distance'], data2[ci], 'b-', label='data', linewidth=1)
        plt.scatter(X_test, y_pred_test.detach().numpy(), label='Test values', c='red', s=1)
        plt.scatter(X_train, y_pred_train.detach().numpy(), label='Train values', c='black', s=6)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"{ci} Fit de Morse pour {i}%")
        plt.legend()
        plt.show()
        '''
        print("De:{:.2f}".format(De_fit), "a:{:.2f}".format(a_fit), "re:{:.2f}".format(re_fit), "v:{:.2f}".format(v_fit))

        # Enregistrer les résultats pour chaque `i` dans les dictionnaires
        all_rmse_results[i] = rmse_Ts
        all_mse_results[i] = mse_Ts
        
        all_test_results[i] = y_pred_test
        all_dvr_results[i] = y_pred_dvr
        
        
    # Les résultats Dataframes
    
    all_rmse_results_df0 = {k: v.item() for k, v in all_rmse_results.items()}
    all_rmse_results_df = pd.DataFrame(all_rmse_results_df0, index=[0])
    
    
    all_dvr_results_df = pd.concat([all_dvr_results['iter1'],all_dvr_results['iter2'],all_dvr_results['iter3'],all_dvr_results['iter4'], all_dvr_results['iter5'], all_dvr_results['iter6'], all_dvr_results['iter7'], all_dvr_results['iter8']], axis=1)
    all_dvr_results_df.columns = ['iter1','iter2','iter3','iter4', 'iter5', 'iter6', 'iter7', 'iter8']
        
    all_test_results_df0 = {key: pd.DataFrame(value.numpy(), columns=[key]) for key, value in all_test_results.items()}
    all_test_results_df = pd.concat(all_test_results_df0.values(), axis=1)

      
    # Enregistrer les résultats dans un fichier CSV pour chaque CI
    output_path_rmse = fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Morse_rmse_results.csv'
    #all_rmse_results_df.to_csv(output_path_rmse, index=False)
    
    output_path = fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Morse_dvr_results.csv'
    #all_dvr_results_df.to_csv(output_path, index=False)
    
    output_path = fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Morse_test_results.csv'
    #all_test_results_df.to_csv(output_path, index=False)
    
    












