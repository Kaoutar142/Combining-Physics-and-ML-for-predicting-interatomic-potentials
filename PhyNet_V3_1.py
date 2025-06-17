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
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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
        print(f"Itération {i}: Total points = {total_points_to_select}, Additional points = {additional_points_to_select}")
        
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
            # Plots tests
            plt.plot(data2['Distance'].to_numpy(), data2['CI1'].to_numpy(), 'b-', label='data', linewidth=1)
            plt.scatter(selected_data['Distance'].to_numpy(), selected_data['CI1'].to_numpy(), label='Test values',c='red', s=4)
            plt.show()
            '''
            
        # Sélectionner le reste des points pour le test
        selected_data_test = data[~data.index.isin(selected_data.index)]
        '''
        # Plots tests
        plt.plot(data2['Distance'], data2['CI1'], 'b-', label='data', linewidth=1)
        plt.scatter(selected_data_test['Distance'], selected_data_test['CI1'], label='Test values',c='red', s=4)
        plt.scatter(selected_data['Distance'], selected_data['CI1'], label='Train values',c='green', s=6)
        
        plt.show()
        '''
        # Stocker le dataset correspondant à l'itération actuelle dans le dictionnaire
        selected_data_dict[f"iter{i}"] = selected_data.copy()
        selected_data_test_dict[f"iter{i}"] = selected_data_test.copy()
        
    return selected_data_dict, selected_data_test_dict


# Sélection des données en étapes et stockage dans un dictionnaire
data_selected_dict, selected_data_test_dict = select_data_in_steps(data2, 5, 8)

# Data DVR
dvr00 = np.arange(0.5,9,0.01)
dvr0 = torch.tensor(dvr00, dtype=torch.float32)
dvr = dvr0.unsqueeze(1)
                                                                                                                                                                    
# Définition des classes des Modèles
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def morse_torch(x, De, a, re, v):
    return (De * ((1 - torch.exp(-a * (x - re)))**2)) + v


class MorseModel(torch.nn.Module):
    def __init__(self):
        super(MorseModel, self).__init__()
        self.De = torch.nn.Parameter(torch.tensor(4, dtype=torch.float32))
        self.a = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        self.re = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        
    def forward(self, x):
        return morse_torch(x, self.De, self.a, self.re, self.v)

class NetEstimator(nn.Module):
    def __init__(self):
        super(NetEstimator, self).__init__()      
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            #nn.ReLU(),
            nn.Linear(12, 1),)
        # Appliquer l'initialisation Xavier
        self.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
                
    def forward(self,x):
        return self.net(x)
    

class Combined(nn.Module):
    def __init__(self, model_phy, model_aug):
        super(Combined, self).__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        # Initialiser l comme un paramètre apprenable
        self.k = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.l = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
    def forward(self, x):
        pred_phy = self.model_phy(x)
        pred_aug = self.model_aug(x)
        
        combined_pred = ((self.k)*pred_phy) + ((self.l)*pred_aug)
        
        return combined_pred, pred_aug, pred_phy     


# Fonction de perte personnalisée
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def custom_loss(output_combined, target, output_aug, output_phy):
    # Instancier MSELoss
    mse_loss_fn = torch.nn.MSELoss()
    
    # Erreur de prédiction combinée
    total_loss = mse_loss_fn(output_combined, target)
    
    # Calculer l'erreur du modèle physique
    error_phy = mse_loss_fn(output_phy , target)
    error_aug = mse_loss_fn(output_aug , target)
    
    return total_loss, error_phy, error_aug


# Fix random seeds for reproducibility
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# Function Train et Evaluate
# -----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

def train_and_evaluate(seed):
    set_seed(seed)
    model_aug = NetEstimator()
    model_phy = MorseModel()
    model = Combined(model_phy=model_phy, model_aug=model_aug)        
    
    optimizer = optim.Adam(model.parameters(), lr=0.000009)
    
    plot_interval = 200000
    
    # Training loop
    for epoch in range(1400000):
        model.train()
        
        # Prédiction du modèle
        y_pred, y_pred_aug, y_pred_phy = model(X_train)
        
        # Calcul de la perte
        loss_total, error_phy, error_aug = custom_loss(y_pred, y_train, y_pred_aug, y_pred_phy)
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        #if epoch % 200 == 0:
            #print(f'Epoch {epoch}, Loss: {loss_total.item():.10f}')
        
            
        '''
        if epoch % plot_interval == 0:

            y_pred_train0 = model(X_train)
            y_pred_train = pd.DataFrame(y_pred_train0[0].detach().numpy())
            
            y_pred_test0 = model(X_test)
            y_pred_test = pd.DataFrame(y_pred_test0[0].detach().numpy())
            
            y_pred_dvr0 = model(dvr)
            y_pred_dvr = pd.DataFrame(y_pred_dvr0[0].detach().numpy())
            
            # Calculate RMSE
            mse_Ts = ((y_pred_test[0] - y_data_test) ** 2).mean()
            rmse_Ts = np.sqrt(mse_Ts)
            
            mse_Tr = ((y_pred_train[0] - y_data) ** 2).mean()
            
            # Plot Test
            plt.text(0.4, 0.7, 'MSE Ts: {:.10f}'.format(mse_Ts), transform=plt.gca().transAxes, fontsize=10, color='blue')
            plt.text(0.4, 0.65, 'MSE Tr: {:.10f}'.format(mse_Tr), transform=plt.gca().transAxes, fontsize=10, color='blue')
            plt.plot(data2['Distance'].to_numpy(), data2[ci].to_numpy(), 'b-', label='data', linewidth=1)
            plt.scatter(x_data_test, y_pred_test[0].to_numpy(), label='Test values',c='red', s=2)
            plt.scatter(x_data, y_pred_train[0].to_numpy(), label='Train values',c='black', s=6)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f"PhyNet Test {ci} - train {i} - seed {seed} points")
            plt.legend()
            plt.show()
        '''
        
    print("\nParamètres finaux du modèle physique après entraînement :")
    print(f"De = {model.model_phy.De.item():.4f}, a = {model.model_phy.a.item():.4f}, "
      f"re = {model.model_phy.re.item():.4f}, v = {model.model_phy.v.item():.4f}")
    print("")

    
    y_pred_train0 = model(X_train)
    y_pred_train = pd.DataFrame(y_pred_train0[0].detach().numpy())
    
    y_pred_test0 = model(X_test)
    y_pred_test = pd.DataFrame(y_pred_test0[0].detach().numpy())
    
    y_pred_dvr0 = model(dvr)
    y_pred_dvr = pd.DataFrame(y_pred_dvr0[0].detach().numpy())
    
    # Calculate RMSE
    mse_Ts = ((y_pred_test[0] - y_data_test) ** 2).mean()
    rmse_Ts = np.sqrt(mse_Ts)
    
    mse_Tr = ((y_pred_train[0] - y_data) ** 2).mean()
    
    # Plot Test
    plt.text(0.4, 0.7, 'MSE Ts: {:.10f}'.format(mse_Ts), transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.text(0.4, 0.65, 'MSE Tr: {:.10f}'.format(mse_Tr), transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.plot(data2['Distance'].to_numpy(), data2[ci].to_numpy(), 'b-', label='data', linewidth=1)
    plt.scatter(x_data_test, y_pred_test[0].to_numpy(), label='Test values',c='red', s=2)
    plt.scatter(x_data, y_pred_train[0].to_numpy(), label='Train values',c='black', s=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"PhyNet Test {ci} - train {i} - seed {seed} points")
    plt.legend()
    plt.show()
        
    return mse_Ts, rmse_Ts, y_pred_test[0],  y_pred_dvr[0]

# Training PhysiNet
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# Boucle pour chaque CI (CI1 à CI6)
for ci in ['CI2'] : #['CI1', 'CI2', 'CI3', 'CI4', 'CI5', 'CI6']

    # Initialiser les dictionnaires pour stocker les résultats
    all_rmse_results = {}
    all_mse_results = {}
    
    all_y_pred_test_values = {}
    all_y_pred_dvr_values = {}
    
    keys_to_use = list(data_selected_dict.keys())[0:8]
    
    for i in keys_to_use:
        
    #if len(data_selected_dict) >= 4:

        #i = list(data_selected_dict.keys())[4]
        
        # Data Train
        x_data = data_selected_dict[i]['Distance']
        y_data = data_selected_dict[i][ci]
        
        x_data = x_data.reset_index(drop=True)
        y_data = y_data.reset_index(drop=True)
            
        # Data Test
        x_data_test = (selected_data_test_dict[i]['Distance'])
        y_data_test = (selected_data_test_dict[i][ci])
        
        x_data_test = x_data_test.reset_index(drop=True)
        y_data_test = y_data_test.reset_index(drop=True)
    
        # Réseau de neuronnes
        # ----------------------------------------------------------------------------
            
        # Convert to 2D PyTorch tensors
        X_train = torch.tensor(x_data, dtype=torch.float32).reshape(-1, 1)
        y_train = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(x_data_test, dtype=torch.float32).reshape(-1, 1)
        y_test = torch.tensor(y_data_test, dtype=torch.float32).reshape(-1, 1)
                    
        # Loop over multiple seeds
        seeds = (2,4,10,22,23,28,29,30,32,42,52)
        
        rmse_results = []
        mse_results = []
        
        y_pred_test_values = {}
        y_pred_dvr_values = {}
        
        for seed in seeds:
            
            mse, rmse, y_pred_test, y_pred_dvr = train_and_evaluate(seed)
            
            mse_results.append({'seed': seed, 'mse': mse})
            rmse_results.append({'seed': seed, 'rmse': rmse})
            
            y_pred_test_values[seed] = y_pred_test.tolist()
            y_pred_dvr_values[seed] = y_pred_dvr.tolist()
            
        # Enregistrer les résultats pour chaque `i` dans les dictionnaires
        all_rmse_results[i] = rmse_results
        all_mse_results[i] = mse_results 
        
        all_y_pred_test_values[i] = y_pred_test_values
        all_y_pred_dvr_values[i] = y_pred_dvr_values
        
        y_pred_test_df = pd.DataFrame.from_dict(all_y_pred_test_values[i], orient='index').transpose()
        y_pred_dvr_df = pd.DataFrame.from_dict(all_y_pred_dvr_values[i], orient='index').transpose()
        
        #y_pred_test_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\physinet\{ci}\physinet2_test_results_{i}.csv', index=False)
        #y_pred_dvr_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\physinet\{ci}\physinet2_dvr_results_{i}.csv', index=False)
    
    # Outputs
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    
    # Enregistrer les résultats sous forme CSV pour chaque `i`
    
    seeds = [entry['seed'] for entry in all_rmse_results[list(all_rmse_results.keys())[0]]]
    
    rmse_df = pd.DataFrame({'seed': seeds})
    test_df = pd.DataFrame({'seed': seeds})
    dvr_df = pd.DataFrame({'seed': seeds})
    
    for i in all_rmse_results:
        rmse_values = [entry['rmse'] for entry in all_rmse_results[i]]
        rmse_df[f'iter_{i}'] = rmse_values
    #rmse_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\physinet\{ci}\physinet2_rmse_results.csv', index=False)


