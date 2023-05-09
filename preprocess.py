# Open admissions.csv and convert to a dataframe
# Clean the data
# Save the dataframe as a pickle file

import pandas as pd
import numpy as np
import pickle

def preprocessAdmissions():
    # Read in the csv file
    df = pd.read_csv('data/admissions.csv')

    # Choose 1000 random patients 
    df = df.sample(n=100000, random_state=42)

    # Create a new dataframe with only one column for subject_id
    master_df = df[['subject_id']]
    master_df = master_df.drop_duplicates()

    # Save the subject_id dataframe as a pickle file
    with open('data/master_df.pickle', 'wb') as f:
        pickle.dump(master_df, f)

def preprocessDiagnoses():
    # Get the master_df dataframe
    with open('data/master_df.pickle', 'rb') as f:
        master_df = pickle.load(f)

    # Read in the csv file
    df = pd.read_csv('data/diagnoses_icd.csv')

    # Filter the dataframe to only include the subject_id in the master_df
    df = df[df['subject_id'].isin(master_df['subject_id'])]

    # Add a column called 'Hypertension' to master_df
    master_df['Hypertension'] = 0
    # If the patient has a diagnosis of hypertension with diagnosis code 4019, 4011, or I10, set the value to 1
    master_df.loc[master_df['subject_id'].isin(df[df['icd_code'].isin(['4019', '4011', 'I10'])]['subject_id']), 'Hypertension'] = 1

    # Add a column called 'Hypercholesterolemia' to master_df
    master_df['Hypercholesterolemia'] = 0
    # If the patient has a diagnosis of hypercholesterolemia with diagnosis code 2720, set the value to 1
    master_df.loc[master_df['subject_id'].isin(df[df['icd_code'].isin(['2720'])]['subject_id']), 'Hypercholesterolemia'] = 1

    # Add a column called 'Atherosclerosis' to master_df
    master_df['Atherosclerosis'] = 0
    # If the patient has a diagnosis of atherosclerosis with diagnosis code 41401, set the value to 1
    master_df.loc[master_df['subject_id'].isin(df[df['icd_code'].isin(['41401'])]['subject_id']), 'Atherosclerosis'] = 1

    # Dump the master_df as a pickle file
    with open('data/master_df.pickle', 'wb') as f:
        pickle.dump(master_df, f)


    print(master_df.head())

    # Count and print number of patients with athlerosclerosis, hypertension, and hypercholesterolemia
    print('Number of patients with atherosclerosis: {}'.format(master_df['Atherosclerosis'].sum()))
    print('Number of patients with hypertension: {}'.format(master_df['Hypertension'].sum()))
    print('Number of patients with hypercholesterolemia: {}'.format(master_df['Hypercholesterolemia'].sum()))

# def preprocessLabEvents():
#     # Get the master_df
#     with open('data/master_df.pickle', 'rb') as f:
#         master_df = pickle.load(f)

#     # Read in the csv file
#     df = pd.read_csv('data/labevents.csv')

#     print(df.head())

preprocessAdmissions()
preprocessDiagnoses()