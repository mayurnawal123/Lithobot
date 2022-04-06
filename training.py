import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import requests
import zipfile
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

############### Loading data  ##################

def load_data():
    # ## Fetching the data from github
    # url = 'https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/raw/master/lithology_competition/data/train.zip'
    # r = requests.get(url, allow_redirects=True)
    # if not os.path.exists('Data'):
    #     os.mkdir('Data')
    # open('Data/train.zip', 'wb').write(r.content)

    # ## Unziping zip file
    # with zipfile.ZipFile('Data/train.zip', 'r') as zip_ref:
    #     zip_ref.extractall('Data/')

    ## Reading the data
    raw_data = pd.read_csv("Data/train.csv", sep = ';')
    lithology_keys = {30000: 'Sandstone',
                    65030: 'Sandstone/Shale',
                    65000: 'Shale',
                    80000: 'Marl',
                    74000: 'Dolomite',
                    70000: 'Limestone',
                    70032: 'Chalk',
                    88000: 'Halite',
                    86000: 'Anhydrite',
                    99000: 'Tuff',
                    90000: 'Coal',
                    93000: 'Basement'}


    raw_data = raw_data.replace({"FORCE_2020_LITHOFACIES_LITHOLOGY": lithology_keys})

    raw_data.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'Lithofacies'}, inplace=True)

    raw_data.rename(columns={'FORCE_2020_LITHOFACIES_CONFIDENCE':'Confidence'}, inplace=True)

    data_headers = [   'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF',
        'DTC', 'SP', 'BS', 'ROP', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC',
        'ROPA', 'RXO', 'Lithofacies', ]

    return raw_data[data_headers]

# def exploratory_data_analysis():
#     pass


def preprocessing(data):
    ############################ Thresholding  ############################
    data['CALI'] = np.array(np.where((data['CALI'] <= 0) | (data['CALI'] >=30), np.nan, data['CALI'])) 
    data['RSHA'] = np.array(np.where((data['RSHA'] <= 0) | (data['RSHA'] >=100), np.nan, data['RSHA']))
    data['RMED'] = np.array(np.where((data['RMED'] <= 0) | (data['RMED'] >=100), np.nan, data['RMED']))
    data['RDEP'] =np.array(np.where((data['RDEP'] <= 0) | (data['RDEP'] >= 100 ), np.nan, data['RDEP']  ))
    data['RHOB'] = np.array(np.where((data['RHOB'] <= 1) | (data['RHOB'] >= 3), np.nan, data['RHOB'])  )
    data['GR'] = np.array(np.where((data['GR'] <= 0) | (data['GR'] >=200), np.nan, data['GR']))
    data['SGR'] = np.array(np.where((data['SGR'] <= 0) | (data['SGR'] >=200), np.nan, data['SGR']))
    data['NPHI'] = np.array(np.where((data['NPHI'] <= 0) | (data['NPHI'] >= 0.6), np.nan, data['NPHI'])  )
    data['PEF'] =np.array(np.where((data['PEF'] <=0 ) | (data['PEF'] >=10 ), np.nan, data['PEF']  ))
    data['DTC'] =np.array(np.where((data['DTC'] <= 0) | (data['DTC'] >=300 ), np.nan, data['DTC']  ))
    data['SP'] =np.array(np.where((data['SP'] <= -200) | (data['SP'] >= 200), np.nan, data['SP']  ))
    data['BS'] =np.array(np.where((data['BS'] <= 0) | (data['BS'] >=30 ), np.nan, data['BS']  ))
    data['ROP'] =np.array(np.where((data['ROP'] <= 0) | (data['ROP'] >=60 ), np.nan, data['ROP']  ))
    data['DTS'] =np.array(np.where((data['DTS'] <= 0) | (data['DTS'] >=300 ), np.nan, data['DTS']  ))
    data['DCAL'] = np.array(np.where((data['DCAL'] <= 0) | (data['DCAL'] >=2), np.nan, data['DCAL']))
    data['DRHO'] =np.array(np.where((data['DRHO'] <= -0.1) | (data['DRHO'] >=0.1 ), np.nan, data['DRHO']  ))
    data['MUDWEIGHT'] =np.array(np.where((data['MUDWEIGHT'] <= 0) | (data['MUDWEIGHT'] >=1.5 ), np.nan, data['MUDWEIGHT']  ))
    data['RMIC'] =np.array(np.where((data['RMIC'] <= 0) | (data['RMIC'] >=30 ), np.nan, data['RMIC']  ))
    data['ROPA'] =np.array(np.where((data['ROPA'] <= 0) | (data['ROPA'] >=60 ), np.nan, data['ROPA']  ))
    data['RXO'] =np.array(np.where((data['RXO'] <= 0) | (data['RXO'] >=20 ), np.nan, data['RXO']  ))

    ####################### Removing NaN values  #############################
    def replace_nan(x):
        if pd.isnull(x):
            return 999999
        else:
            return x
    
    for header in data.columns[:-1]:
        data[header] = data[header].apply(replace_nan) # Can it be replaces inplace?
    
    # print('Preprocessing done')
    return data

ct = {
        'comb1' :[   'CALI', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'SP', 'BS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'RXO', ],
        'comb2' : [   'CALI', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 
                'RMIC', 'RXO' ],
        'comb3' : [   'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'DTS', 'DCAL', 
                'DRHO', 'MUDWEIGHT', 'RMIC', 'RXO'  ],
        'comb4' : [   'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS', 
                'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO' ]
    }


def train(data,header,model_name):
    a = 0 
    b = 100
    test_sz = 9.99

    X_train, X_test, Y_train, Y_test = train_test_split(data[header], data['Lithofacies'], random_state = int(b),
                                                     test_size= (test_sz/10), stratify = data['Lithofacies'])
    clf = RandomForestClassifier(n_estimators = 200, max_depth=70,  min_samples_split=2, min_samples_leaf=1,max_features=10, random_state=0)
    clf.fit(X_train, Y_train)
    ## Saving model
    if not os.path.exists('Models'):
        os.mkdir('Models')
    outfile = open('Models/'+model_name,'ab')
    pickle.dump(clf,outfile)
    outfile.close()

    predicted_output = clf.predict(X_train)
    true_lithofacies = Y_train 

    # cm = confusion_matrix(true_lithofacies, predicted_output)
    # cmp = ConfusionMatrixDisplay(cm) 
    # fig, ax = plt.subplots(figsize=(10,10))
    # cmp.plot(ax=ax)
    # plt.show()

    # print("\nTraining Accuracy:", accuracy_score(true_lithofacies, predicted_output))
    # print("\n -------------Training Classification Report-------------\n")
    # print(classification_report(true_lithofacies, predicted_output))
    return clf
    # predicted_output = clf.predict(X_test)
    # true_lithofacies = Y_test 

    # cm = confusion_matrix(true_lithofacies, predicted_output)
    # cmp = ConfusionMatrixDisplay(cm) # , display_labels=np.arange(25))
    # fig, ax = plt.subplots(figsize=(10,10))
    # cmp.plot(ax=ax)

    # print(" test Accuracy:", accuracy_score(true_lithofacies, predicted_output))
    # print("\n ------------- Test Classification Report-------------\n")
    # print(classification_report(true_lithofacies, predicted_output))



def train_model1(data):
    # print('Training started')
    comb1 = ct['comb1']
    return train(data,comb1,'Model_1')

def feature_importance(clf,ct,comb,plot=True):
    importances = clf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    keys = ct[comb]
    values = importances
    imp_dict = {}
    for i,col in enumerate(keys):
        imp_dict[col] = values[i]
    if plot:
        plt.figure(figsize=(15,6))
        plt.title('Feature Importance')
        plt.xlabel('Logs')
        plt.ylabel('Relative importance')
        plt.bar(range(len(ct[comb])), importances[sorted_indices], align='center')
        plt.xticks(range(len(ct[comb])), np.array(ct[comb])[sorted_indices], rotation=90)
        plt.tight_layout()
        plt.show()
    if not os.path.exists('Models'):
        os.mkdir('Models')
    outfile = open('Models/imp_'+comb,'ab')
    pickle.dump(imp_dict,outfile)
    outfile.close()
    return imp_dict


    ######################## 4 datasets for each combinations #########################
    # 
    # comb2 = [   'CALI', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 
    #         'RMIC', 'RXO', ]
    # comb3 = [   'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'DTS', 'DCAL', 
    #         'DRHO', 'MUDWEIGHT', 'RMIC', 'RXO',  ]
    # comb4 = data_headers[:-1]

    ####################### Training combination-1  ###########################
    
    
    ## reading model
    # file = open('Models/'+model_namemodel_name,'rb')
    # clf = pickle.load(file)
    # file.close()
    
    
if __name__ == '__main__':
    data = load_data()
    data = preprocessing(data)
    clf = train_model1(data)
    feature_importance(clf,ct,comb='comb1')