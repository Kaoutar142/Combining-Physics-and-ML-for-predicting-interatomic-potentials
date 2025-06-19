# Hybrid Physics-ML Models for Interatomic Potentials

This repository contains the source code used in the research article [Combining Physics and Machine Learning: Hybrid Models for predicting interatomic potentials]. 
The study explores various hybrid approaches combining physics-based models and machine learning to predict interatomic potential energy curves. 
The models are evaluated on their ability to reproduce the ground and first excited electronic states of diatomic molecules, with a particular focus on Franck-Condon factors and MSE performance.

## Repository Structure

The repository includes the implementation of the following models:

- Aphynity.py – Implementation of the **APHYNITY** model.
- PhysiNet.py – Implementation of the **PhysiNet** model.
- Sequential.py – Implementation of the **Sequential Phy-ML** model.
- Fit_morse_V3.py – Fitting procedure for the **physics-based Morse potential** model.
- Plot_RMSE_4-8.py – Script to compute and plot the **RMSE** of predictions across models.
- FFC_All_seed V1.py – Script for computing **Franck-Condon factors** and evaluating the RMSE between predicted and reference factors across models and seeds.

## Data

The dataset used in this work contains discrete potential energy values as a function of internuclear distances for the **ground** state and the **first seven excited states** of a diatomic molecule (H2).

📁 **data file**:  
Potential_Energy_Curves_H2_GS_ES1-7.csv

