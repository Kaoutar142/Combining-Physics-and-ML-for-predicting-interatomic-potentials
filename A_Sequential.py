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
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

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
            
            # Plots tests
            '''
            plt.plot(data2['Distance'], data2['CI1'], 'b-', label='data', linewidth=1)
            plt.scatter(selected_data['Distance'], selected_data['CI1'], label='Test values',c='red', s=4)
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

                                                                                                                                                                    
# Model Phy
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def morse_torch(x, De, a, re, v):
    return (De * ((1 - torch.exp(-a * (x - re)))**2)) + v

# Modèle PyTorch pour ajuster les paramètres
class MorseModel(torch.nn.Module):
    def __init__(self):
        super(MorseModel, self).__init__()
        self.De = torch.nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.a = torch.nn.Parameter(torch.tensor(3, dtype=torch.float32))
        self.re = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float32))
        self.v = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        return morse_torch(x, self.De, self.a, self.re, self.v)

# Model Aug
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
            nn.Linear(12, 1),
        )

    def forward(self, x):
        return self.net(x)

# Model combiné
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

def custom_loss(output_combined, target, output_aug, output_phy):
    
    mse_loss_fn = torch.nn.MSELoss()    
    total_loss = mse_loss_fn(output_combined, target)
    
    error_phy = mse_loss_fn(output_phy, target)
    error_aug = mse_loss_fn(output_aug, error_phy)
    
    return total_loss, error_phy, error_aug


# Function to train the model and compute RMSE for given seed
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def train_and_evaluate(seed):
    set_seed(seed)

    # Entraînement du modèle physique
    model_phy = MorseModel()
    optimizer_phy = optim.Adam(model_phy.parameters(), lr=0.01)
    error_phy_history = []
    
    for epoch in range(100000):  # Ajustez le nombre d'époques selon vos besoins
        y_pred_phy = model_phy(X_train)
        loss_phy = torch.nn.functional.mse_loss(y_pred_phy, y_train)
    
        optimizer_phy.zero_grad()
        loss_phy.backward()
        optimizer_phy.step()
        
        #print(f"Epoch {epoch} Phy - Loss: {loss_phy.item()} - CI: {ci}")
        
        error_phy_history.append(loss_phy.item())
        
    
    # Paramètres ajustés
    De_fit = model_phy.De.item()
    a_fit = model_phy.a.item()
    re_fit = model_phy.re.item()
    v_fit = model_phy.v.item()

    #print("De:{:.2f}".format(De_fit) , "a:{:.2f}".format(a_fit), "re:{:.2f}".format(re_fit), "v:{:.2f}".format(v_fit))

    # Plot Modèle Phy
    
    y_pred_train_phy = morse_torch(X_train, De_fit, a_fit, re_fit, v_fit)
    y_pred_test_phy = morse_torch(X_test, De_fit, a_fit, re_fit, v_fit)

    mse_Ts_phy = ((y_pred_test_phy - y_test)**2).mean()
    rmse_Ts_phy = np.sqrt(mse_Ts_phy)

    #plt.text(0.3, 0.8, 'MSE Train: {:.6f}'.format(mse_Tr), transform=plt.gca().transAxes, fontsize=10, color='blue')
    #plt.text(0.3, 0.75, 'RMSE Train: {:.6f}'.format(rmse_Tr), transform=plt.gca().transAxes, fontsize=10, color='blue')
    '''
    plt.text(0.3, 0.7, 'MSE Test: {:.10f}'.format(mse_Ts_phy), transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.text(0.3, 0.65, 'RMSE Test: {:.10f}'.format(rmse_Ts_phy), transform=plt.gca().transAxes, fontsize=10, color='blue')

    plt.plot(data2['Distance'], data2['CI2'], 'b-', label='data', linewidth=1)

    plt.scatter(X_test, y_pred_test_phy, label='Test values',c='red', s=1)
    plt.scatter(X_train, y_pred_train_phy, label='Train values',c='black', s=6)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(" Fit de morse pour 5% ")
    plt.legend()
    plt.show()
    '''
    # Entrainement Modèle combiné
    
    model_aug = NetEstimator()
    model_combined = Combined(model_phy=model_phy, model_aug=model_aug)
    
    for param in model_combined.model_phy.parameters():
        param.requires_grad = False
    
    optimizer_combined = optim.Adam(model_combined.parameters(), lr=lr)
    
    loss_total_history = []
    error_aug_history = []
    best_mse = np.inf
    
    for epoch in range(n_epochs):  # Ajustez le nombre d'époques selon vos besoins
        y_pred, y_pred_aug, y_pred_phy = model_combined(X_train)
        loss_total, error_phy, error_aug = custom_loss(y_pred, y_train, y_pred_aug, y_pred_phy)
        
        #print(f"Epoch {epoch} combined - Loss: {loss_total.item()} - CI: {ci}")
        
        optimizer_combined.zero_grad()
        loss_total.backward()
        optimizer_combined.step()
        
        # Track the best model based on MSE
        if loss_total.item() < best_mse:
            best_mse = loss_total.item()
            best_weights = model_combined.state_dict()
            
        loss_total_history.append(loss_total.item())
        error_aug_history.append(error_aug.item())
    
    '''    
    # Tracer les erreurs et la perte totale
    plt.figure(figsize=(10, 7))
    #plt.plot(loss_total_history, label='Erreur Totale', color='blue')
    #plt.plot(error_phy_history, label='Erreur PHY', color='green')
    plt.plot(error_aug_history, label='Erreur AUG', color='red')
    plt.xlabel('Époque')
    plt.ylabel('Erreur')
    plt.title(f'Courbes d\'erreur au cours de l\'entraînement pour {i} et seed {seed}')
    plt.legend()
    plt.show()
    '''
    print("\nParamètres finaux du modèle physique après entraînement :")
    print(f"De = {model_combined.model_phy.De.item():.4f}, a = {model_combined.model_phy.a.item():.4f}, "
      f"re = {model_combined.model_phy.re.item():.4f}, v = {model_combined.model_phy.v.item():.4f}")
    print("")
    
    # Evaluation on test set
    model_combined.eval()
    model_combined.load_state_dict(best_weights)
    
    y_pred_train0 = model_combined(X_train)
    y_pred_train = pd.DataFrame(y_pred_train0[0].detach().numpy())
    
    y_pred_test0 = model_combined(X_test)
    y_pred_test = pd.DataFrame(y_pred_test0[0].detach().numpy())
    
    y_pred_dvr0 = model_combined(dvr)
    y_pred_dvr = pd.DataFrame(y_pred_dvr0[0].detach().numpy())
    
    t1 = pd.DataFrame(y_pred_test0[1].detach().numpy())
    t2 = pd.DataFrame(y_pred_test0[2].detach().numpy())
    
    # Calculate RMSE
    mse_Ts = ((y_pred_test[0] - y_data_test) ** 2).mean()
    rmse_Ts = np.sqrt(mse_Ts)
    
    mse_Tr = ((y_pred_train[0] - y_data) ** 2).mean()
    rmse_Tr = np.sqrt(mse_Tr)
    
    '''
    # Plot Test
    plt.text(0.4, 0.7, 'MSE Ts: {:.10f}'.format(mse_Ts), transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.text(0.4, 0.65, 'MSE Tr: {:.10f}'.format(mse_Tr), transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.plot(data2['Distance'], data2[ci], 'b-', label='data', linewidth=1)
    plt.scatter(x_data_test, y_pred_test[0], label='Test values',c='red', s=2)
    plt.scatter(x_data, y_pred_train[0], label='Train values',c='black', s=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"New Test {ci} - train {i} - seed {seed} points")
    plt.legend()
    plt.show() 
    '''
    return mse_Ts, rmse_Ts, y_pred_test[0],  y_pred_dvr[0]

# Fix random seeds
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Fix random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
            
# Training New
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Définir les valeurs spécifiques de lr et lambda_penalty pour chaque itération
lr_values = {
    'iter1': 0.001,
    'iter2': 0.001,
    'iter3': 0.001,
    'iter4': 0.001,
    'iter5': 0.00092,
    'iter6': 0.00092,
    'iter7': 0.00092,
    'iter8': 0.00092
}


epochs = {
    'iter1': 100000,
    'iter2': 100000,
    'iter3': 100000,
    'iter4': 100000,
    'iter5': 400000,
    'iter6': 400000,
    'iter7': 400000,
    'iter8': 400000
}


for ci in ['CI2']: #, 'CI3', 'CI4', 'CI5', 'CI6']:

    # Initialiser les dictionnaires pour stocker les résultats
    all_rmse_results = {}
    all_mse_results = {}
    
    all_y_pred_test_values = {}
    all_y_pred_dvr_values = {}
    
    # Boucle pour itérer à travers chaque pourcentage
    #for i in data_selected_dict:
    
    keys_to_use = list(data_selected_dict.keys())[0:8]
    
    for i in keys_to_use:
        
        # Définir les valeurs de lr et lambda_penalty pour cette itération
        lr = lr_values[i]
        epoch = epochs[i]
        
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
        
        dvr00 = np.arange(0.5,9,0.01)
        dvr0 = torch.tensor(dvr00, dtype=torch.float32)
        dvr = dvr0.unsqueeze(1)        
        
        n_epochs = epoch
        
        # Loop over multiple seeds
        seeds = (2,4,10,22,23,28,29,30,32,42,52) # 10 different seeds
        
        rmse_results = []
        mse_results = []
        
        y_pred_test_values = {}
        y_pred_dvr_values = {}
        
        
        for seed in seeds:
            
            mse, rmse, y_pred_test, y_pred_dvr = train_and_evaluate(seed)
            
            mse_results.append({'seed': seed, 'mse': mse})
            rmse_results.append({'seed': seed, 'rmse': rmse})
            
            y_pred_test_values[seed] = y_pred_test.tolist()  # Convertir en liste pour éviter les erreurs de type
            y_pred_dvr_values[seed] = y_pred_dvr.tolist()
            
        # Enregistrer les résultats pour chaque `i` dans les dictionnaires
        all_rmse_results[i] = rmse_results
        all_mse_results[i] = mse_results 
    
        all_y_pred_test_values[i] = y_pred_test_values
        all_y_pred_dvr_values[i] = y_pred_dvr_values
        
        y_pred_test_df = pd.DataFrame.from_dict(all_y_pred_test_values[i], orient='index').transpose()
        y_pred_dvr_df = pd.DataFrame.from_dict(all_y_pred_dvr_values[i], orient='index').transpose()
        
        #y_pred_test_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Test_new\New2_test_results_{i}.csv', index=False)
        #y_pred_dvr_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Test_new\New2_dvr_results_{i}.csv', index=False)
        
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
    #rmse_df.to_csv(fr'C:\Users\kaoutar\Desktop\Projet 2\output\V3\{ci}\Test_new\New2_rmse_results.csv', index=False)




