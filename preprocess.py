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
    df = df.sample(n=10000, random_state=42)

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

    # Count and print number of patients with athlerosclerosis, hypertension, and hypercholesterolemia
    print('Number of patients with atherosclerosis: {}'.format(master_df['Atherosclerosis'].sum()))
    print('Number of patients with hypertension: {}'.format(master_df['Hypertension'].sum()))
    print('Number of patients with hypercholesterolemia: {}'.format(master_df['Hypercholesterolemia'].sum()))

def preprocessPatients():
    # Get the master_df dataframe
    with open('data/master_df.pickle', 'rb') as f:
        master_df = pickle.load(f)

    # Read in the csv file
    df = pd.read_csv('data/patients.csv')

    # Filter the dataframe to only include the subject_id in the master_df
    df = df[df['subject_id'].isin(master_df['subject_id'])]

    # Add column called 'Male' to master_df
    master_df['Male'] = 0

    # If the patient has gender 'M', set the value to 1
    master_df.loc[master_df['subject_id'].isin(df[df['gender'].isin(['M'])]['subject_id']), 'Male'] = 1

    # Add a column called 'Female' to master_df
    master_df['Female'] = 0

    # If patient has gender 'F', set the value to 1
    master_df.loc[master_df['subject_id'].isin(df[df['gender'].isin(['F'])]['subject_id']), 'Male'] = 1

    # Add columns called 'Age <40', 'Age 40-59', 'Age 60-79', and 'Age 80+' to master_df
    master_df['Age <40'] = 0
    master_df['Age 40-59'] = 0
    master_df['Age 60-79'] = 0
    master_df['Age 80+'] = 0

    # If the patient is less than 40 years old, set the value to 1
    # If the patient is between 40 and 59 years old, set the value to 1
    # If the patient is between 60 and 79 years old, set the value to 1
    # If the patient is 80 years old or older, set the value to 1
    # Look at anchor_age column and classify patients into age groups
    master_df.loc[master_df['subject_id'].isin(df[df['anchor_age'] < 40]['subject_id']), 'Age <40'] = 1
    master_df.loc[master_df['subject_id'].isin(df[df['anchor_age'].between(40, 59, inclusive=True)]['subject_id']), 'Age 40-59'] = 1
    master_df.loc[master_df['subject_id'].isin(df[df['anchor_age'].between(60, 79, inclusive=True)]['subject_id']), 'Age 60-79'] = 1
    master_df.loc[master_df['subject_id'].isin(df[df['anchor_age'] >= 80]['subject_id']), 'Age 80+'] = 1

    # Dump the master_df as a pickle file
    with open('data/master_df.pickle', 'wb') as f:
        pickle.dump(master_df, f)


    