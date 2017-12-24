# -*- coding: utf-8 -*-

def getFileName(fileName):
    if fileName == 'vote':
        return "dataset/vote2.csv"
    elif fileName == 'toy':
        return "dataset/toy.csv"
    elif fileName == 'mushroom':
        return "dataset/mushroom1.csv"
    elif fileName == 'weather':
        return "dataset/weathernominal.csv"
    else:
        return ""
    
    
    
    
def getFileHeader(fileName):
     if fileName == 'vote':
         return ['handicapped-infants','water-project-cost-sharing','adoption-of-the-budget-resolution','physician-fee-freeze',
                'el-salvador-aid','religious-groups-in-schools','anti-satellite-test-ban','aid-to-nicaraguan-contras',
                'mx-missile','immigration','synfuels-corporation-cutback','education-spending','superfund-right-to-sue',
                'crime','duty-free-exports','export-administration-act-south-africa','class']
     elif fileName == 'toy':
         return ["COLOR", "DIAMETER", "CLASS"]
     elif fileName == 'mushroom':
         return ['cap-shape', 'cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
                'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
                'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-colo','population',
                'habitat','class']
     elif fileName == 'weather':
         return['outlook','Temperature ','Humidity ','Windy','Play']
     else:
         return []
    
