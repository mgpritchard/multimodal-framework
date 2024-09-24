#!/bin/bash 
cd /users/p/pritcham/multimodal-opt/mm-opt-experiments/

python testFusion.py featlevel LOO server 100 False
zip lit_data_expts/jeong/results/featlevel_jointselect.zip lit_data_expts/jeong/results/Fusion_CSP/LOO/*
rm lit_data_expts/jeong/results/Fusion_CSP/LOO/*
