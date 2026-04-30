import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def add_features(df):
    df = df.copy().sort_values(['patient_id', 'diabetic_year'])
 
    # Yearly change in BMI and weight
    df['bmi_change']    = df.groupby('patient_id')['BMI_in_kg_per_m2'].diff().fillna(0)
    df['weight_change'] = df.groupby('patient_id')['weight_in_kg'].diff().fillna(0)
 
    # Score from presence of comorbidities
    df['comorbidity_score'] = (
        df['has_hypertension'] + df['has_heart_disease'] +
        df['urinary_infection'] + df['takes_pain_killer']
    )
 
    # Healthy lifestyle score
    df['lifestyle_score'] = (
        df['diet_regulation']      + df['takes_diabetes_medicine'] +
        df['takes_insulin']        + df['sufficient_sleep']        +
        df['sufficient_water_consumption']     + df['walk_regularly']
    )
 
    # Risk score from smoking and tobacco
    df['risk_behaviour'] = df['smoking_habit'] + df['takes_tobacco']
 
    # No of years of having CKD so far
    df['ckd_years_so_far'] = df.groupby('patient_id')['CKD'].cumsum() - df['CKD']
 
    return df

def rename_columns(df):
    # Dictionary of new column names for efficient usage
    new_column_names = {
        'Patient ID':'patient_id', 
        'Gender (M-male, F-female)': 'gender',
        'Job (1-normal, 2-intermediate, 3-heavy)': 'job',
        'Family Background of Diabetes (1-yes, 0-no)': 'family_diabetic_history',
        'Height (cm)': 'height_in_cm',
        'Diabetic Year': 'diabetic_year', 
        'Age': 'age_in_years', 
        'Average Age': 'average_age', 
        'Weight (kg)': 'weight_in_kg',
        'Average Weight (kg)': 'average_weight', 
        'BMI': 'BMI_in_kg_per_m2', 
        'Follow suggested Diet (1-yes, 0-no)': 'diet_regulation',
        'Take Medicine for Diabetes (1-yes, 0-no)': 'takes_diabetes_medicine',
        'Take Insulin (1-yes, 0-no)': 'takes_insulin',
        'Hypertension (1-yes, 0-no)': 'has_hypertension',
        'Heart Disease (1-yes, 0-no)': 'has_heart_disease', 
        'Sleep (1-sufficient, 0-insufficient)': 'sufficient_sleep',
        'Water Consumption (1-sufficient, 0-insufficient)': 'sufficient_water_consumption',
        'Smoke (1-yes, 0-no)': 'smoking_habit', 
        'Zarda, Betel Leaf (1-yes, 0-no)': 'takes_tobacco',
        'Walk Regularly (1-yes, 0-no)': 'walk_regularly', 
        'Urination Properly (1-yes, 0-no)': 'proper_urination',
        'Urinary Infection (1-yes, 0-no)': 'urinary_infection', 
        'Pain killer (1-yes, 0-no)': 'takes_pain_killer',
        'Calorie Intake (per day)': 'calorie_intake_per_day', 
        'CKD (1-yes, 0-no)': 'CKD'
    }
    # Rename columns of the dataframe using the new column names
    df = df.rename(columns=new_column_names)
    return df

def split_data(df):
    all_patients = df['patient_id'].unique()
 
    train_patients, temp  = train_test_split(all_patients, test_size=0.30, random_state=42)
    val_patients,   test_patients = train_test_split(temp, test_size=0.50, random_state=42)
    
    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df   = df[df['patient_id'].isin(val_patients)].copy()
    test_df  = df[df['patient_id'].isin(test_patients)].copy()
    
    print(f"\nAfter Split — Train: {train_df['patient_id'].nunique()} patients | "
        f"Val: {val_df['patient_id'].nunique()} patients | "
        f"Test: {test_df['patient_id'].nunique()} patients")
    
    return train_df, val_df, test_df

def main():
    df = pd.read_csv('./dataset/DiabeticCKD_dataset.csv')

    # Display first 10 rows from dataset
    print(f"First 10 rows of the dataset:\n{df.head(10)}")

    # Display last 10 rows from dataset
    print(f"\nLast 10 rows of the dataset:\n{df.tail(10)}")

    # Dataset Information (column names, data types)
    print(f"\nDataset Information:\n{df.info()}")

    # Dataset Description 
    # Statistical description of each column like count, average, standard deviation, min, max, etc.
    print(f"\nDataset Description:\n{df.describe().T}")

    # Missing values in each column
    print(f"\nMissing values in each column:\n{df.isnull().sum()}")

    # Shape of the Dataset 
    print(f"\nDataset shape: {df.shape}")

    # Number of Patients
    print(f"\nPatients: {df['Patient ID'].nunique()}")

    # Save original dataset for later use (default: deep = True)
    df_original = df.copy()

    # Rename columns for efficient usage
    df = rename_columns(df)
    print(f"\nRenamed columns:\n{df.columns}")

    # Remove already aggregated columns
    df = df.drop(columns=['average_age', 'average_weight'])
    print(f"\nColumns after dropping aggregated columns:\n{df.columns}")

    # Binary-encoding the feature 'Gender'
    print(f"\nBinary-encoding the feature 'Gender'...")
    df["gender"] = df["gender"].map({"M": 1, "F": 0})
    print(f"\nUnique values in 'gender' column after encoding: {df['gender'].unique()}")

    # Add new features to the dataset
    df = add_features(df)
    print(f"\nColumns after adding new features {len(df.columns)}:\n{df.columns}")

    df_clean = df.copy()

    print(f"\nSplitting data into train, validation and test sets...")
    train_df, val_df, test_df = split_data(df_clean)

    # Save cleaned datasets and split datasets for later use
    print(f"\nSaving cleaned datasets and split datasets for later use...")
    np.save('./dataset/clean_data.npy', df_clean)
    np.save('./dataset/train_df.npy', train_df)
    np.save('./dataset/val_df.npy', val_df)
    np.save('./dataset/test_df.npy', test_df)
    pd.Series(df_clean.columns).to_csv('./dataset/column_names.csv', index=False)
    print(f"\nData preprocessing completed and saved successfully!")

if __name__ == "__main__":
    main()
