#!/bin/bash 
cd /users/p/pritcham/multimodal-opt/mm-opt-experiments/

python testFusion.py just_emg LOO server 100 False
zip lit_data_expts/jeong/results/EMG_only.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*

python testFusion.py just_eeg LOO server 100 False
zip lit_data_expts/jeong/results/EEG_only.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*

python testFusion.py featlevel LOO server 100 False
zip lit_data_expts/jeong/results/featlevel_separateselect.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*
