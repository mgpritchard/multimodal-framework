#!/bin/bash 
cd /users/p/pritcham/multimodal-opt/mm-opt-experiments/

python testFusion.py decision LOO server 1 False
zip lit_data_expts/jeong/results/decision.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*

python testFusion.py hierarchical LOO server 1 False
zip lit_data_expts/jeong/results/hierarchical_prob.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*

python testFusion.py hierarchical_inv LOO server 1 False
zip lit_data_expts/jeong/results/hierarchical_inv_prob.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*

python testFusion.py just_emg LOO server 1 False
zip lit_data_expts/jeong/results/EMG_only.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*

python testFusion.py just_eeg LOO server 1 False
zip lit_data_expts/jeong/results/EEG_only.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*

python testFusion.py featlevel LOO server 1 False
zip lit_data_expts/jeong/results/featlevel_jointselect.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*