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
from torch.optim.lr_scheduler import StepLR
                                                                                                                                                             
# Datas
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Charger les données
data0 = pd.read_csv(r"C:\Users\kaoutar\Desktop\pec_sigmag.txt", header=None, delimiter=" ")
data0.rename(columns={0:'Distance',1:'CI1',2:'CI2',3:'CI3',4:'CI4',5:'CI5',6:'CI6',7:'CI7',8:'CI8'}, inplace=True)
data2 = data0[data0["Distance"] < 20]

dvr00 = np.arange(0.5,9,0.01)
dvr0 = torch.tensor(dvr00, dtype=torch.float32)
dvr = dvr0.unsqueeze(1)

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
            
            # Plots tests
            
            #plt.plot(data2['Distance'], data2['CI2'], 'b-', label='data', linewidth=1)
            #plt.scatter(selected_data['Distance'], selected_data['CI2'], label='Test values',c='red', s=4)
            #plt.show()
            
            
        # Sélectionner le reste des points pour le test
        selected_data_test = data[~data.index.isin(selected_data.index)]

        # Stocker le dataset correspondant à l'itération actuelle dans le dictionnaire
        selected_data_dict[f"iter{i}"] = selected_data.copy()
        selected_data_test_dict[f"iter{i}"] = selected_data_test.copy()
        
    return selected_data_dict, selected_data_test_dict


# Sélection des données en étapes et stockage dans un dictionnaire
data_selected_dict, selected_data_test_dict = select_data_in_steps(data2, 5, 8)

                                                                                                                                                                    
# Model Phy
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Définir la fonction de Morse dans PyTorch

def morse_torch(x, De, a, re, v):
    return (De * ((1 - torch.exp(-a * (x - re)))**2)) + v


# Modèle PyTorch pour ajuster les paramètres
class MorseModel(torch.nn.Module):
    def __init__(self):
        super(MorseModel, self).__init__()
        self.De = torch.nn.Parameter(torch.tensor(4, dtype=torch.float32))
        self.a = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        self.re = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
        
    def forward(self, x):
        return morse_torch(x, self.De, self.a, self.re, self.v)

# Model aug
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
  
class NetEstimator(nn.Module):
    def __init__(self):
        super(NetEstimator, self).__init__()      
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
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
  
# Fusion des deux modèle
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class Combined(nn.Module):
    def __init__(self, model_phy, model_aug):
        super(Combined, self).__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug

        
    def forward(self, x):
        pred_phy = self.model_phy(x)
        pred_aug = self.model_aug(x)
        
        combined_pred = pred_phy + pred_aug
        
        
        return combined_pred, pred_aug, pred_phy   

# Fonction de perte personnalisée
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def custom_loss(output_combined, target, output_aug, lambda_combined, lambda_penalty):
        mse_loss_fn = torch.nn.MSELoss()
        Real_loss = mse_loss_fn(output_combined, target)
        
        error_phy = mse_loss_fn(output_combined - output_aug , target)
        error_aug = mse_loss_fn(output_aug , target)
        total_loss = lambda_combined * Real_loss + lambda_penalty * error_phy
        
        
        return total_loss, Real_loss, error_phy, error_aug

# Function to train the model and compute RMSE for given seed
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def train_and_evaluate(seed, lr, lambda_penalty):
    set_seed(seed)
    model_aug = NetEstimator()
    model_phy = MorseModel()
    model = Combined(model_phy=model_phy, model_aug=model_aug)        
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_mse = np.inf   # Init to infinity
    best_weights = None
    history = []
    error_phy_history = []
    error_aug_history = []
    Real_history = []
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        
        # Prédiction du modèle
        y_pred, y_pred_aug, y_pred_phy = model(X_train)
        
        
        # Calcul de la perte
        loss_total, Real_loss, error_phy, error_aug = custom_loss(y_pred, y_train, y_pred_aug,
                                                       lambda_combined=1, lambda_penalty=lambda_penalty)
            
        #if epoch % 300 == 0:
            #print(f'CI {ci}, Epoch {epoch}, Total Loss: {loss_total.item()}, Real Loss: {Real_loss.item()}')

        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Rétropropagation
        loss_total.backward()
                    
        # Mise à jour des paramètres
        optimizer.step()
    
        
        # Track the best model based on MSE
        if loss_total.item() < best_mse:
            best_mse = loss_total.item()
            best_weights = model.state_dict()

        # Optionally append loss to history
        history.append(loss_total.item())
        Real_history.append(Real_loss.item())
        error_phy_history.append(error_phy.item())
        error_aug_history.append(error_aug.item())
       
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Real Loss')
    plt.plot(error_phy_history, label='Physical Error')
    plt.plot(error_aug_history, label='Augmented Error')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')  # Mettre l'échelle de y en log
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    print("\nParamètres finaux du modèle physique après entraînement :")
    print(f"De = {model.model_phy.De.item():.4f}, a = {model.model_phy.a.item():.4f}, "
      f"re = {model.model_phy.re.item():.4f}, v = {model.model_phy.v.item():.4f}")
    print("")
    
    # Evaluation on test set
    model.eval()
    model.load_state_dict(best_weights)
    
    y_pred_train0 = model(X_train)
    y_pred_train = pd.DataFrame(y_pred_train0[0].detach().numpy())
    
    y_pred_test0 = model(X_test)
    y_pred_test = pd.DataFrame(y_pred_test0[0].detach().numpy())
    
    y_pred_dvr0 = model(dvr)
    y_pred_dvr = pd.DataFrame(y_pred_dvr0[0].detach().numpy())
    
    t1 = pd.DataFrame(y_pred_test0[1].detach().numpy())
    t2 = pd.DataFrame(y_pred_test0[2].detach().numpy())
    
    # Calculate RMSE
    mse_Ts = ((y_pred_test[0] - y_data_test) ** 2).mean()
    rmse_Ts = np.sqrt(mse_Ts)
    
    mse_Tr = ((y_pred_train[0] - y_data) ** 2).mean()
    rmse_Tr = np.sqrt(mse_Tr)
    
    
    # Plot Test
    plt.text(0.4, 0.7, 'MSE Ts: {:.10f}'.format(mse_Ts), transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.text(0.4, 0.65, 'MSE Tr: {:.10f}'.format(mse_Tr), transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.plot(data2['Distance'].to_numpy(), data2[ci].to_numpy(), 'b-', label='data', linewidth=1)
    plt.scatter(x_data_test, y_pred_test[0].to_numpy(), label='Test values',c='red', s=2)
    plt.scatter(x_data, y_pred_train[0].to_numpy(), label='Train values',c='black', s=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Aph Test {ci} - train {i} - seed {seed} points")
    plt.legend()
    plt.show()
    
    '''
    # Plot contribution
    plt.plot(data2['Distance'], data2[ci], 'b-', label='data', linewidth=1)
    plt.plot(X_test, t1[0], 'r-', label='NN', linewidth=1)
    plt.plot(X_test, t2[0], 'g-', label='Phy', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Aph {ci} - train 20% points - contributions")
    plt.legend()
    plt.show()
    '''
    return mse_Ts, rmse_Ts, y_pred_test[0],  y_pred_dvr[0]


# Fix random seeds
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Training Aphynity
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Définir les valeurs spécifiques de lr et lambda_penalty pour chaque itération
lr_values = {
    'iter1': 0.0015,
    'iter2': 0.00098,
    'iter3': 0.001,
    'iter4': 0.00098,
    'iter5': 0.0007,
    'iter6': 0.0007,
    'iter7': 0.0007,
    'iter8': 0.0007
}

lambda_penalty_values = {
    'iter1': 2,
    'iter2': 6,
    'iter3': 6,
    'iter4': 6,
    'iter5': 3,
    'iter6': 3,
    'iter7': 3,
    'iter8': 4
}

epochs = {
    'iter1': 500000,
    'iter2': 400000,
    'iter3': 500000,
    'iter4': 400000,
    'iter5': 1200000,
    'iter6': 1200000,
    'iter7': 1200000,
    'iter8': 1400000
}
'''
# Définir les valeurs spécifiques de lr et lambda_penalty pour chaque itération
lr_values = {
    'iter1': 0.001,
    'iter2': 0.00098,
    'iter3': 0.00098,
    'iter4': 0.0009,
    'iter5': 0.0006,
    'iter6': 0.0006,
    'iter7': 0.00062,
    'iter8': 0.0007
}

lambda_penalty_values = {
    'iter1': 1,
    'iter2': 6,
    'iter3': 6,
    'iter4': 6,
    'iter5': 6,
    'iter6': 6,
    'iter7': 6,
    'iter8': 4
}

epochs = {
    'iter1': 200000,
    'iter2': 200000,
    'iter3': 400000,
    'iter4': 400000,
    'iter5': 700000,
    'iter6': 700000,
    'iter7': 600000,
    'iter8': 1400000
}
'''

# Boucle pour chaque CI (CI1 à CI6)
for ci in ['CI1']:
    
    # Initialiser les dictionnaires pour stocker les résultats
    all_rmse_results = {}
    all_mse_results = {}
    
    all_y_pred_test_values = {}
    all_y_pred_dvr_values = {}
    
    keys_to_use = list(data_selected_dict.keys())[0:8]
    
    for i in keys_to_use:
        
        # Définir les valeurs de lr et lambda_penalty pour cette itération
        lr = lr_values[i]
        lambda_penalty = lambda_penalty_values[i]
        n_epochs = epochs[i]

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
            
        # Convert to 2D PyTorch tensors
        X_train = torch.tensor(x_data, dtype=torch.float32).reshape(-1, 1)
        y_train = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1)
        X_test = torch.tensor(x_data_test, dtype=torch.float32).reshape(-1, 1)
        y_test = torch.tensor(y_data_test, dtype=torch.float32).reshape(-1, 1)
    
       
        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error  
        #n_epochs = epoch   # number of epochs to run
        
        rmse_results = []
        mse_results = []
        
        y_pred_test_values = {}
        y_pred_dvr_values = {}

        # Loop over multiple seeds
        seeds = (2,4,10,22,23,28,29,30,32,42,52)
         
        for seed in seeds:
            
            mse, rmse, y_pred_test2, y_pred_dvr2 = train_and_evaluate(seed,lr,lambda_penalty)
            
            mse_results.append({'seed': seed, 'mse': mse})
            rmse_results.append({'seed': seed, 'rmse': rmse})
            
            y_pred_test_values[seed] = y_pred_test2.tolist()  # Convertir en liste pour éviter les erreurs de type
            y_pred_dvr_values[seed] = y_pred_dvr2.tolist()
         
            
        # Enregistrer les résultats pour chaque `i` dans les dictionnaires
        all_rmse_results[i] = rmse_results
        all_mse_results[i] = mse_results 
        
        all_y_pred_test_values[i] = y_pred_test_values
        all_y_pred_dvr_values[i] = y_pred_dvr_values
        
        y_pred_test_df = pd.DataFrame.from_dict(all_y_pred_test_values[i], orient='index').transpose()
        y_pred_dvr_df = pd.DataFrame.from_dict(all_y_pred_dvr_values[i], orient='index').transpose()
        
        #y_pred_test_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Test\AphNv4_test_results_{i}.csv', index=False)
        #y_pred_dvr_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Test\AphNv4_dvr_results_{i}.csv', index=False)
    
    # Outputs
    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    
    # Enregistrer les résultats sous forme CSV pour chaque `i`
    
    seeds = [entry['seed'] for entry in all_rmse_results[list(all_rmse_results.keys())[0]]]
    rmse_df = pd.DataFrame({'seed': seeds})
    
    for i in all_rmse_results:
        rmse_values = [entry['rmse'] for entry in all_rmse_results[i]]
        rmse_df[f'iter_{i}'] = rmse_values
    #rmse_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Test\AphNv4_rmse_results.csv', index=False)
    
  
    