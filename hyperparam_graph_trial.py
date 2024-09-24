# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:54:02 2023

@author: pritcham
"""


import re
from collections.abc import Iterable
import graphviz
####https://github.com/opizzato/dict_to_digraph/blob/main/dict_to_digraph.py

import os
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin/"

def dict_to_digraph(input_dict, params):

    edges = dict_to_edges(input_dict, params)

    dot = graphviz.Digraph(
        format=params.get('format', 'svg'),
        graph_attr=params.get('graph_attr', {}),
    )
    for s,d in edges:
        labels = [text_to_label(t, params) for t in [s,d]]
        dot.edge(*labels)

    nodes = list(set([n for n,_ in edges] + [n for _,n in edges]))
    for n in nodes:
        attr = node_to_attr(n, params)
        label = text_to_label(n, params)
        dot.node(label, **attr)

    return dot


def text_to_label(text, params):

    for t in params.get('text_transforms', []):
        text = t(text)

    return text

def node_to_attr(node, params):

    attr = {}
    for regexp, a in params.get('node_attrs', []):
        if re.match(regexp, node):
            attr.update(a)

    return attr


def key_value_to_sources(k, v, params):

    sources = []

    for fk in params.get('edge_keys', []):

        if isinstance(fk, str):
            if re.match(fk, k):
                if isinstance(v, str):
                    sources.append(v)

        if isinstance(fk, dict):
            for dfk, fks in fk.items():
                if re.match(dfk, k):
                    if isinstance(v, dict):
                        for fk in fks:
                            for sk,sv in v.items():
                                if re.match(fk, sk):
                                    if isinstance(sv, str):
                                        sources.append(sv)
                    if isinstance(v, Iterable):
                        for vv in v:
                            if isinstance(vv, dict):
                                for fk in fks:
                                    for sk,sv in vv.items():
                                        if re.match(fk, sk):
                                            if isinstance(sv, str):
                                                sources.append(sv)
    return sources


def dict_to_edges(d, params):

    edges = []
    if isinstance(d, dict):
        for k,v in d.items():
            if isinstance(v, dict):
                for vk, vv in v.items():
                    sources = key_value_to_sources(vk, vv, params)
                    for s in sources:
                        edges.append([s,k])

    return edges

###https://github.com/opizzato/dict_to_digraph/blob/main/dict_to_digraph.py

if 0:    
    emg_dict={
            'rf':{'emg_model_type':'RF',
                     'n_trees':'10 - 100, q5',
                     'max_depth':5,#scope.int(hp.quniform('emg.RF.maxdepth',2,5,q=1)),
                     #integerising search space https://github.com/hyperopt/hyperopt/issues/566#issuecomment-549510376
                     },
            'knn':{'emg_model_type':'kNN',
                     'knn_k':'1-25',
                     },
             'lda': {'emg_model_type':'LDA',
                     'LDA_solver':['svd','lsqr','eigen'], #removed lsqr due to LinAlgError: SVD did not converge in linear least squares but readding as this has not repeated
                     'shrinkage':'0.0 - 1.0',
                     },
             'qda': {'emg_model_type':'QDA', #emg qda reg 0.3979267 actually worked well!! for withinppt
                     'regularisation':[0.0 , 1.0], #https://www.kaggle.com/code/code1110/best-parameter-s-for-qda/notebook
                     },
              'gnb':   {'emg_model_type':'gaussNB',
                     'smoothing':'log 1e-9 - 1',
                     },
     #               {'emg_model_type':'SVM',    #SKL SVC likely unviable, excessively slow
      #               'svm_C':hp.uniform('emg.svm.c',0.1,100), #use loguniform?
       #              },
                    }
    params=['smoothing','emg_model_type','n_trees','max_depth','knn_k','LDA_solver','shrinkage','regularisation']
    #emg_dot=dict_to_digraph(emg_dict,{'edge_keys':params})
    ##doctest_mark_exe()
    #emg_dot.render(view=True)
    
    sysdot=graphviz.Digraph('space2',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
    
    emgdot=graphviz.Digraph('emg2',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
    emgdot.node('emgtop','EMG Model')
    emgdot.node('emgrf','RF')
    emgdot.node('emgrfntrees','n_trees',style='solid')
    emgdot.node('emgrfntrees.vals','1 - 100 (steps 5)',style='solid')
    emgdot.node('emgknn','KNN')
    emgdot.node('emg.knn.k','k',style='solid')
    emgdot.node('emg.knn.k.vals','1 - 25',style='solid')
    emgdot.node('emglda','LDA')
    emgdot.node('emg.lda.solver','solver',style='solid')
    #below is actual
    emgdot.node('emg.lda.solver.vals','SVD\nLSQR\nEigen',style='solid')
    emgdot.node('emg.lda.shrinkage','shrinkage',style='solid')
    emgdot.node('emg.lda.shrinkage.vals','0.0 - 1.0',style='solid')
    #below is theoretical
    #emgdot.node('emg.lda.solver.svd','SVD',shape='box')
    #emgdot.node('emg.lda.solver.nonSVD','LSQR\nEigen',shape='box')
    #emgdot.node('emg.lda.shrinkage','shrinkage',shape='box')
    #emgdot.node('emg.lda.shrinkage.vals','0.0 - 1.0',shape='box')
    emgdot.node('emgqda','QDA')
    emgdot.node('emg.qda.regularisation','regularisation',style='solid')
    emgdot.node('emg.qda.regularisation.vals','0.0 - 1.0',style='solid')
    emgdot.node('emggnb','GNB')
    emgdot.node('emg.gnb.smoothing','smoothing',style='solid')
    emgdot.node('emg.gnb.smoothing.vals','1e-9 - 1 (log)',style='solid')
    emgdot.node('emgsvm','SVM')
    emgdot.node('emg.svm.C','C',style='solid')
    emgdot.node('emg.svm.C.vals','0.1 - 100 (log)',style='solid')
    emgdot.node('emg.svm.gamma','gamma',style='solid')
    emgdot.node('emg.svm.gamma.vals','0.01 - 1 (log)',style='solid')
    emgdot.edges([('emgtop','emgrf'),('emgrf','emgrfntrees'),('emgrfntrees','emgrfntrees.vals'),
                  ('emgtop','emgknn'),('emgknn','emg.knn.k'),('emg.knn.k','emg.knn.k.vals'),
                  #below is actual
                  ('emgtop','emglda'),('emglda','emg.lda.solver'),('emg.lda.solver','emg.lda.solver.vals'),
                                      ('emglda','emg.lda.shrinkage'),('emg.lda.shrinkage','emg.lda.shrinkage.vals'),
                  #below is theoretical
                  #('emgtop','emglda'),('emglda','emg.lda.solver'),('emg.lda.solver','emg.lda.solver.svd'),
                  #    ('emg.lda.solver','emg.lda.solver.nonSVD'),
                  #        ('emg.lda.solver.nonSVD','emg.lda.shrinkage'),('emg.lda.shrinkage','emg.lda.shrinkage.vals'),
                  ('emgtop','emgqda'),('emgqda','emg.qda.regularisation'),('emg.qda.regularisation','emg.qda.regularisation.vals'),
                  ('emgtop','emggnb'),('emggnb','emg.gnb.smoothing'),('emg.gnb.smoothing','emg.gnb.smoothing.vals'),
                  ])
    emgdot.edge('emgtop','emgsvm',style='dashed',label='Bespoke only')
    emgdot.edges([('emgsvm','emg.svm.C'),('emg.svm.C','emg.svm.C.vals'),
                  ('emgsvm','emg.svm.gamma'),('emg.svm.gamma','emg.svm.gamma.vals')])
    #sysdot.subgraph(emgdot)
    
    eegdot=graphviz.Digraph('eeg2',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
    eegdot.node('eegtop','EEG Model')
    eegdot.node('eegrf','RF')
    eegdot.node('eegrfntrees','trees',style='solid')
    eegdot.node('eegrfntrees.vals','1 - 100 (steps 5)',style='solid')
    eegdot.node('eegknn','KNN')
    eegdot.node('eeg.knn.k','k',style='solid')
    eegdot.node('eeg.knn.k.vals','1 - 25',style='solid')
    eegdot.node('eeglda','LDA')
    eegdot.node('eeg.lda.solver','solver',style='solid')
    #below is actual
    eegdot.node('eeg.lda.solver.vals','SVD\nLSQR\nEigen',style='solid')
    eegdot.node('eeg.lda.shrinkage','shrinkage',style='solid')
    eegdot.node('eeg.lda.shrinkage.vals','0.0 - 1.0',style='solid')
    #below is theoretical
    #eegdot.node('eeg.lda.solver.svd','SVD',shape='box')
    #eegdot.node('eeg.lda.solver.nonSVD','LSQR\nEigen',shape='box')
    #eegdot.node('eeg.lda.shrinkage','shrinkage',shape='box')
    #eegdot.node('eeg.lda.shrinkage.vals','0.0 - 1.0',shape='box')
    eegdot.node('eegqda','QDA')
    eegdot.node('eeg.qda.regularisation','regularisation',style='solid')
    eegdot.node('eeg.qda.regularisation.vals','0.0 - 1.0',style='solid')
    eegdot.node('eeggnb','GNB')
    eegdot.node('eeg.gnb.smoothing','smoothing',style='solid')
    eegdot.node('eeg.gnb.smoothing.vals','1e-9 - 1 (log)',style='solid')
    eegdot.node('eegsvm','SVM')
    eegdot.node('eeg.svm.C','C',style='solid')
    eegdot.node('eeg.svm.C.vals','0.1 - 100 (log)',style='solid')
    eegdot.node('eeg.svm.gamma','gamma',style='solid')
    eegdot.node('eeg.svm.gamma.vals','0.01 - 1 (log)',style='solid')
    eegdot.edges([('eegtop','eegrf'),('eegrf','eegrfntrees'),('eegrfntrees','eegrfntrees.vals'),
                  ('eegtop','eegknn'),('eegknn','eeg.knn.k'),('eeg.knn.k','eeg.knn.k.vals'),
                  #below is actual
                  ('eegtop','eeglda'),('eeglda','eeg.lda.solver'),('eeg.lda.solver','eeg.lda.solver.vals'),
                                      ('eeglda','eeg.lda.shrinkage'),('eeg.lda.shrinkage','eeg.lda.shrinkage.vals'),
                  #below is theoretical
                  #('eegtop','eeglda'),('eeglda','eeg.lda.solver'),('eeg.lda.solver','eeg.lda.solver.svd'),
                  #    ('eeg.lda.solver','eeg.lda.solver.nonSVD'),
                  #        ('eeg.lda.solver.nonSVD','eeg.lda.shrinkage'),('eeg.lda.shrinkage','eeg.lda.shrinkage.vals'),
                  ('eegtop','eegqda'),('eegqda','eeg.qda.regularisation'),('eeg.qda.regularisation','eeg.qda.regularisation.vals'),
                  ('eegtop','eeggnb'),('eeggnb','eeg.gnb.smoothing'),('eeg.gnb.smoothing','eeg.gnb.smoothing.vals'),
                  ])
    eegdot.edge('eegtop','eegsvm',style='dashed',label='Bespoke only')
    eegdot.edges([('eegsvm','eeg.svm.C'),('eeg.svm.C','eeg.svm.C.vals'),
                  ('eegsvm','eeg.svm.gamma'),('eeg.svm.gamma','eeg.svm.gamma.vals')])
    #sysdot.subgraph(eegdot)
    
    decdot=graphviz.Digraph('decfus2',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
    decdot.node('dectop','Decision Fusion Algorithm')
    decdot.node('dec.mean','Mean')
    decdot.node('dec.EMG','EMG-biased')# Average')
    decdot.node('dec.EEG','EEG-biased')# Average')
    decdot.node('dec.tune','Tunable Average')#'Parameter-Weighted Average')
    decdot.node('dec.tune.weight','EEG weighting',style='solid')
    decdot.node('dec.tune.weight.vals','0.0 - 100.0',style='solid')
    decdot.node('dec.max','Maximum')
    decdot.node('dec.svm','Linear SVM')
    decdot.node('dec.svm.C','C',style='solid')
    decdot.node('dec.svm.C.vals','0.01 - 100\n(log scaled)',style='solid')
    decdot.node('dec.lda','LDA')
    decdot.node('dec.lda.solver','solver',style='solid')
    decdot.node('dec.lda.solver.vals','SVD\nLSQR\nEigen',style='solid')
    decdot.node('dec.lda.shrinkage','shrinkage',style='solid')
    decdot.node('dec.lda.shrinkage.vals','0.0 - 1.0',style='solid')
    decdot.node('dec.rf','RF')
    decdot.node('dec.rf.ntrees','n_trees',style='solid')
    decdot.node('dec.rf.ntrees.vals','1 - 100\n(steps of 5)',style='solid')
    decdot.edges([('dectop','dec.mean'),
                  ('dectop','dec.EMG'),
                  ('dectop','dec.EEG'),
                  ('dectop','dec.tune'),('dec.tune','dec.tune.weight'),('dec.tune.weight','dec.tune.weight.vals'),
                  ('dectop','dec.max'),
                  ('dectop','dec.svm'),('dec.svm','dec.svm.C'),('dec.svm.C','dec.svm.C.vals'),
                  ('dectop','dec.lda'),('dec.lda','dec.lda.solver'),('dec.lda.solver','dec.lda.solver.vals'),
                                      ('dec.lda','dec.lda.shrinkage'),('dec.lda.shrinkage','dec.lda.shrinkage.vals'),
                  ('dectop','dec.rf'),('dec.rf','dec.rf.ntrees'),('dec.rf.ntrees','dec.rf.ntrees.vals'),
                  ])
    #sysdot.subgraph(decdot)
    
    
    
    MLdot=graphviz.Digraph('ML2',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
    MLdot.node('MLtop','Classifier')
    MLdot.node('MLrf','RF')
    MLdot.node('MLrfntrees','trees',style='solid')
    MLdot.node('MLrfntrees.vals','1 - 100\n(steps of 5)',style='solid')
    MLdot.node('MLknn','KNN')
    MLdot.node('ML.knn.k','k',style='solid')
    MLdot.node('ML.knn.k.vals','1 - 25',style='solid')
    MLdot.node('MLlda','LDA')
    MLdot.node('ML.lda.solver','solver',style='solid')
    #below is actual
    MLdot.node('ML.lda.solver.vals','SVD\nLSQR\nEigen',style='solid')
    MLdot.node('ML.lda.shrinkage','shrinkage',style='solid')
    MLdot.node('ML.lda.shrinkage.vals','0.0 - 1.0',style='solid')
    #below is theoretical
    #MLdot.node('ML.lda.solver.svd','SVD',shape='box')
    #MLdot.node('ML.lda.solver.nonSVD','LSQR\nEigen',shape='box')
    #MLdot.node('ML.lda.shrinkage','shrinkage',shape='box')
    #MLdot.node('ML.lda.shrinkage.vals','0.0 - 1.0',shape='box')
    MLdot.node('MLqda','QDA')
    MLdot.node('ML.qda.regularisation','regularisation',style='solid')
    MLdot.node('ML.qda.regularisation.vals','0.0 - 1.0',style='solid')
    MLdot.node('MLgnb','GNB')
    MLdot.node('ML.gnb.smoothing','smoothing',style='solid')
    MLdot.node('ML.gnb.smoothing.vals','1e-9 - 1\n(log scaled)',style='solid')
    MLdot.node('MLsvm','SVM')
    MLdot.node('ML.svm.C','C',style='solid')
    MLdot.node('ML.svm.C.vals','0.1 - 100\n(log scaled)',style='solid')
    MLdot.node('ML.svm.gamma','gamma',style='solid')
    MLdot.node('ML.svm.gamma.vals','0.01 - 1\n(log scaled)',style='solid')
    MLdot.edges([('MLtop','MLrf'),('MLrf','MLrfntrees'),('MLrfntrees','MLrfntrees.vals'),
                  ('MLtop','MLknn'),('MLknn','ML.knn.k'),('ML.knn.k','ML.knn.k.vals'),
                  #below is actual
                  ('MLtop','MLlda'),('MLlda','ML.lda.solver'),('ML.lda.solver','ML.lda.solver.vals'),
                                      ('MLlda','ML.lda.shrinkage'),('ML.lda.shrinkage','ML.lda.shrinkage.vals'),
                  #below is theoretical
                  #('MLtop','MLlda'),('MLlda','ML.lda.solver'),('ML.lda.solver','ML.lda.solver.svd'),
                  #    ('ML.lda.solver','ML.lda.solver.nonSVD'),
                  #        ('ML.lda.solver.nonSVD','ML.lda.shrinkage'),('ML.lda.shrinkage','ML.lda.shrinkage.vals'),
                  ('MLtop','MLqda'),('MLqda','ML.qda.regularisation'),('ML.qda.regularisation','ML.qda.regularisation.vals'),
                  ('MLtop','MLgnb'),('MLgnb','ML.gnb.smoothing'),('ML.gnb.smoothing','ML.gnb.smoothing.vals'),
                  ])
    MLdot.edge('MLtop','MLsvm',style='dashed',label='Bespoke only')
    MLdot.edges([('MLsvm','ML.svm.C'),('ML.svm.C','ML.svm.C.vals'),
                  ('MLsvm','ML.svm.gamma'),('ML.svm.gamma','ML.svm.gamma.vals')])
    
    
    
    #sysdot.node('sys.root','System')
    sysdot.node('sys.emg','EMG model')
    sysdot.node('sys.eeg','EEG model')
    sysdot.node('sys.feat','Feature-level fused model')
    sysdot.subgraph(MLdot)
    
    #sysdot.edges([('sys.root','sys.emg'),('sys.root','sys.eeg')])
    sysdot.edges([('sys.emg','MLtop'),('sys.eeg','MLtop')])
    #sysdot.edge('sys.root','sys.feat',style='dashed',label='in Feature-level fusion')
    #sysdot.attr(rank='same')
    sysdot.edge('sys.feat','MLtop',style='dotted',label='in Feature-level fusion')
    #sysdot.subgraph(decdot)
    #sysdot.edge('sys.root','dectop',style='dashed',label='in Decision-level fusion')
    
    
    sysdot.render(view=True)
    #decdot.render(view=True)
    
    
    reduceddot=graphviz.Digraph('reduced',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
    
    
    MLdot=graphviz.Digraph('ML_reduced',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
    MLdot.node('MLtop','Classifier')
    MLdot.node('MLrf','RF')
    MLdot.node('MLrfntrees','trees',style='solid')
    MLdot.node('MLrfntrees.vals','10 - 100\n(steps of 5)',style='solid')
    MLdot.node('MLknn','KNN')
    MLdot.node('ML.knn.k','k',style='solid')
    MLdot.node('ML.knn.k.vals','1 - 25',style='solid')
    MLdot.node('MLlda','LDA')
    MLdot.node('ML.lda.solver','solver',style='solid')
    #below is actual
    MLdot.node('ML.lda.solver.vals','SVD\nLSQR\nEigen',style='solid')
    MLdot.node('ML.lda.shrinkage','shrinkage',style='solid')
    MLdot.node('ML.lda.shrinkage.vals','0.0 - 0.5',style='solid')
    #below is theoretical
    #MLdot.node('ML.lda.solver.svd','SVD',shape='box')
    #MLdot.node('ML.lda.solver.nonSVD','LSQR\nEigen',shape='box')
    #MLdot.node('ML.lda.shrinkage','shrinkage',shape='box')
    #MLdot.node('ML.lda.shrinkage.vals','0.0 - 1.0',shape='box')
    MLdot.node('MLqda','QDA')
    MLdot.node('ML.qda.regularisation','regularisation',style='solid')
    MLdot.node('ML.qda.regularisation.vals','0.0 - 1.0',style='solid')
    #MLdot.node('MLgnb','GNB')
    #MLdot.node('ML.gnb.smoothing','smoothing',style='solid')
    #MLdot.node('ML.gnb.smoothing.vals','1e-9 - 1\n(log scaled)',style='solid')
    MLdot.node('MLsvm','SVM')
    MLdot.node('ML.svm.C','C',style='solid')
    MLdot.node('ML.svm.C.vals','0.1 - 100\n(log scaled)',style='solid')
    MLdot.node('ML.svm.gamma','gamma',style='solid')
    MLdot.node('ML.svm.gamma.vals','0.01 - 0.2\n(log scaled)',style='solid')
    MLdot.edges([('MLtop','MLrf'),('MLrf','MLrfntrees'),('MLrfntrees','MLrfntrees.vals'),
                  #below is actual
                  ('MLtop','MLlda'),('MLlda','ML.lda.solver'),('ML.lda.solver','ML.lda.solver.vals'),
                                      ('MLlda','ML.lda.shrinkage'),('ML.lda.shrinkage','ML.lda.shrinkage.vals'),
                  #below is theoretical
                  #('MLtop','MLlda'),('MLlda','ML.lda.solver'),('ML.lda.solver','ML.lda.solver.svd'),
                  #    ('ML.lda.solver','ML.lda.solver.nonSVD'),
                  #        ('ML.lda.solver.nonSVD','ML.lda.shrinkage'),('ML.lda.shrinkage','ML.lda.shrinkage.vals'),
                  ('MLtop','MLqda'),('MLqda','ML.qda.regularisation'),('ML.qda.regularisation','ML.qda.regularisation.vals'),
             #     ('MLtop','MLgnb'),('MLgnb','ML.gnb.smoothing'),('ML.gnb.smoothing','ML.gnb.smoothing.vals'),
                  ])
    MLdot.edge('MLtop','MLsvm',style='dashed',label='Bespoke only')
    MLdot.edges([('MLsvm','ML.svm.C'),('ML.svm.C','ML.svm.C.vals'),
                  ('MLsvm','ML.svm.gamma'),('ML.svm.gamma','ML.svm.gamma.vals')])
    
    MLdot.edge('MLtop','MLknn',style='dashed',label='EMG only')
    MLdot.edges([('MLknn','ML.knn.k'),('ML.knn.k','ML.knn.k.vals')])
    
    
    #sysdot.node('sys.root','System')
    reduceddot.node('sys.emg','EMG model')
    reduceddot.node('sys.eeg','EEG model')
    #sysdot.node('sys.feat','Feature-level fused model')
    reduceddot.subgraph(MLdot)
    
    #sysdot.edges([('sys.root','sys.emg'),('sys.root','sys.eeg')])
    reduceddot.edges([('sys.emg','MLtop'),('sys.eeg','MLtop')])
    #sysdot.edge('sys.root','sys.feat',style='dashed',label='in Feature-level fusion')
    #sysdot.attr(rank='same')
    #sysdot.edge('sys.feat','MLtop',style='dotted',label='in Feature-level fusion')
    #sysdot.subgraph(decdot)
    #sysdot.edge('sys.root','dectop',style='dashed',label='in Decision-level fusion')
    
    
    reduceddot.render(view=True)
    
    
    
    decdot=graphviz.Digraph('decfus_reduced',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
    decdot.node('dectop','Decision Fusion Algorithm')
    decdot.node('dec.mean','Mean')
    decdot.node('dec.EMG','EMG-biased')# Average')
    #decdot.node('dec.EEG','EEG-biased')# Average')
    #decdot.node('dec.tune','Tunable Average')#'Parameter-Weighted Average')
    #decdot.node('dec.tune.weight','EEG weighting',style='solid')
    #decdot.node('dec.tune.weight.vals','0.0 - 100.0',style='solid')
    decdot.node('dec.max','Maximum')
    decdot.node('dec.svm','Linear SVM')
    decdot.node('dec.svm.C','C',style='solid')
    decdot.node('dec.svm.C.vals','0.01 - 100\n(log scaled)',style='solid')
    decdot.node('dec.lda','LDA')
    decdot.node('dec.lda.solver','solver',style='solid')
    decdot.node('dec.lda.solver.vals','SVD\nLSQR\nEigen',style='solid')
    decdot.node('dec.lda.shrinkage','shrinkage',style='solid')
    decdot.node('dec.lda.shrinkage.vals','0.0 - 1.0',style='solid')
    decdot.node('dec.rf','RF')
    decdot.node('dec.rf.ntrees','n_trees',style='solid')
    decdot.node('dec.rf.ntrees.vals','10 - 100\n(steps of 5)',style='solid')
    decdot.edges([('dectop','dec.mean'),
                  ('dectop','dec.EMG'),
    #              ('dectop','dec.EEG'),
    #              ('dectop','dec.tune'),('dec.tune','dec.tune.weight'),('dec.tune.weight','dec.tune.weight.vals'),
                  ('dectop','dec.max'),
                  ('dectop','dec.svm'),('dec.svm','dec.svm.C'),('dec.svm.C','dec.svm.C.vals'),
                  ('dectop','dec.lda'),('dec.lda','dec.lda.solver'),('dec.lda.solver','dec.lda.solver.vals'),
                                      ('dec.lda','dec.lda.shrinkage'),('dec.lda.shrinkage','dec.lda.shrinkage.vals'),
                  ('dectop','dec.rf'),('dec.rf','dec.rf.ntrees'),('dec.rf.ntrees','dec.rf.ntrees.vals'),
                  ])
    
    decdot.render(view=True)



xferdot=graphviz.Digraph('xfer',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
xferdot.node('xfertop','System')
xferdot.node('dectop','Decision Fusion Algorithm')
xferdot.node('dec.mean','Mean')
xferdot.node('dec.EMG','EMG-biased')# Average')
xferdot.node('dec.max','Maximum')
xferdot.node('dec.rf','RF')
xferdot.node('dec.rf.ntrees','n_trees',style='solid')
xferdot.node('dec.rf.ntrees.vals','10 - 100\n(steps of 5)',style='solid')

xferdot.node('MLtop','EMG & EEG\nClassifiers')
xferdot.node('MLrf','RF')
xferdot.node('MLrfntrees','trees',style='solid')
xferdot.node('MLrfntrees.vals','10 - 100\n(steps of 5)',style='solid')

xferdot.node('MLgnb','GNB')
xferdot.node('ML.gnb.smoothing','smoothing',style='solid')
xferdot.node('ML.gnb.smoothing.vals','1e-9 - 0.5 (1 in EEG)\n(log scaled)',style='solid')

xferdot.node('ML.lr','LR')
xferdot.node('ML.lr.C','C',style='solid')
xferdot.node('ML.lr.C.vals','0.01 - 10\n(log scaled)',style='solid')

xferdot.edges([('xfertop','dectop'),('xfertop','MLtop'),
               ('MLtop','MLrf'),('MLrf','MLrfntrees'),('MLrfntrees','MLrfntrees.vals'),
               ('MLtop','MLgnb'),('MLgnb','ML.gnb.smoothing'),('ML.gnb.smoothing','ML.gnb.smoothing.vals'),
               ('MLtop','ML.lr'),('ML.lr','ML.lr.C'),('ML.lr.C','ML.lr.C.vals'),
               ('dectop','dec.mean'),
              ('dectop','dec.EMG'),
              ('dectop','dec.max'),
              ('dectop','dec.rf'),('dec.rf','dec.rf.ntrees'),('dec.rf.ntrees','dec.rf.ntrees.vals'),
              ])

xferdot.render(view=True)






