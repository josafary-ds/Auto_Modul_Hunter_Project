import pandas as pd
import numpy as np
import os
import feets
import warnings


# CREATES A FEETS TABLE WITH ALL NEEDED INFORMATION
def feets_table_generation(input_folder, input_file):
    warnings.simplefilter("ignore")

    # Charges the files
    df_class = pd.read_csv(input_file, sep=';')
    input_dir = os.listdir(input_folder)

    # Creates a longtable with the features from feets
    table = []

    for file in input_dir:
        df = pd.read_csv(f'{input_folder}/{file}', sep=' ', header=None, names=['time', 'flux'])
        time, flux = (df.time, df.flux)
        lc_feets = [time, flux]

        fs = feets.FeatureSpace(data=['magnitude', 'time'], only=['Amplitude', 'Eta_e', 'LinearTrend', 'Mean', 'Meanvariance', 'MedianAbsDev', 'PercentAmplitude', 'PeriodLS', 'Period_fit', 'Psi_CS', 'Psi_eta', 'Std'])
        features, values = fs.extract(*lc_feets)

        id_sec = file.split('.')[0] 
        id_name, sec_name = id_sec.split('_')

        df_1 = pd.DataFrame(data=[[id_name, sec_name, *values]], columns=['id', 'sector']+list(features))
        table.append(df_1)

    df_main = pd.concat(table, ignore_index=True)

    # Creates a second DF with the classes from df_class file and merge to df_main
    old_column_name = df_class.columns[0]
    df_class.rename(columns={old_column_name: 'id'}, inplace=True)

    df_main['id'] = df_main['id'].astype(str)
    df_class['id'] = df_class['id'].astype(str)

    feets_final_table = df_main.merge(df_class, on='id', how='left')

    feets_final_table.to_csv('feets_full_table.csv', sep=';', index=False)



input_folder = input('Insert folder full address: ')
input_file = input('Insert file full address: ')

feets_table_generation(input_folder, input_file)