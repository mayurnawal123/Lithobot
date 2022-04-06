import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle

ct = {
        'comb1' :[   'CALI', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'SP', 'BS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'RXO', ],
        'comb2' : [   'CALI', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 
                'RMIC', 'RXO' ],
        'comb3' : [   'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'DTS', 'DCAL', 
                'DRHO', 'MUDWEIGHT', 'RMIC', 'RXO'  ],
        'comb4' : [   'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS', 
                'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO' ]
    }

def get_comb(fpath='test.csv',sep=';',ct=ct):

    ############## Their header analysis  #####################
    clm = pd.read_csv('Data/train.csv', index_col=0, nrows=0,sep=sep).columns.tolist()

    c = { # preference comb4 to comb1
        'comb1' : len(set(clm).intersection(ct['comb1'])) / len(set(ct['comb1'])),
        'comb2' : len(set(clm).intersection(ct['comb2'])) / len(set(ct['comb2'])),
        'comb3' : len(set(clm).intersection(ct['comb3'])) / len(set(ct['comb3'])),
        'comb4' : len(set(clm).intersection(ct['comb4'])) / len(set(ct['comb4']))
    }

    ## Drop remaining columns from max selected 
    comb = max(zip(c.values(), c.keys()))[1]  ## More improvement can be done by %Null value comparision in data for each combination

    available_clm = []
    for col in ct[comb]:
        if col in clm:
            available_clm.append(col)
    return available_clm,comb

def read_data(comb,available_clm,fname='Data/train.csv',sep=';',ct=ct):
    raw_data = pd.read_csv(fname, sep=sep)
    temp_clm =  [x for x in ct[comb] if x not in available_clm]  # Missing headers in the data for selected combination
    temp_df = pd.DataFrame(columns=temp_clm, index=raw_data.index[:10]) # dataframe for missing logs initialized with Nan
    modified_raw_data = pd.concat([raw_data[available_clm], temp_df], axis=1) # Final data is generated with stacking of available data and missing data
    return modified_raw_data[ct[comb]]

def load_model(comb):
    # reading model
    if comb=='comb4':
        file = open('Models/Model_4','rb')
        clf = pickle.load(file)
        file.close()
    elif comb=='comb3':
        file = open('Models/Model_3','rb')
        clf = pickle.load(file)
        file.close()
    elif comb=='comb2':
        file = open('Models/Model_2','rb')
        clf = pickle.load(file)
        file.close()
    else :
        file = open('Models/Model_1','rb')
        clf = pickle.load(file)
        file.close()
    return clf

def feedback(comb,ct,available_clm):
    if comb=='comb4':
        file = open('Models/imp_comb4','rb')
        imp = pickle.load(file)
        file.close()
    elif comb=='comb3':
        file = open('Models/imp_comb3','rb')
        imp = pickle.load(file)
        file.close()
    elif comb=='comb2':
        file = open('Models/imp_comb2','rb')
        imp = pickle.load(file)
        file.close()
    else :
        file = open('Models/imp_comb1','rb')
        imp = pickle.load(file)
        file.close()
    temp_clm =  [x for x in ct[comb] if x not in available_clm]  # Missing headers in the data for selected combination
    df = pd.DataFrame()
    df['Missing_log'] = imp.keys
    df['Accuracy'] = imp.values
    df = df.loc[df['Missing_log'].isin(temp_clm )]
    print(df)
    return df



available_clm, comb = get_comb(fpath='test.csv',sep=';')
data = read_data(comb, available_clm,sep=';',ct=ct)

clf = load_model(comb)
predictions = clf.predict(data)
df = feedback(comb,ct,available_clm)

