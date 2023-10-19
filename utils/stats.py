import pandas as pd

def count_nan_per_col(df, col_name =""):
    nan_count = df[f'{col_name}'].isna().sum()
    print("There are {} missing values in the column {}".format(nan_count, col_name))

f = "legal_prediction/data/case_cover/case_cover_entities_decision_outcome_9june.csv"
f1 = "legal_prediction/data/case_cover/case_cover_anonymised_extracted_entities.csv"

df = pd.read_csv(f, sep=";")
df1 = pd.read_csv(f1)

###############################################
print(df1.columns)
print(len(df1))
print(df1['extracted_gpe'].head(50))

count_nan_per_col(df1, col_name='Unnamed: 0')
count_nan_per_col(df1, col_name='extracted_dates')
count_nan_per_col(df1, col_name='extracted_gpe')
count_nan_per_col(df1, col_name= 'extracted_org')
count_nan_per_col(df1, col_name= 'public_private_hearing')
count_nan_per_col(df1, col_name='in_chamber_virtual')
count_nan_per_col(df1, col_name='judge_name')
count_nan_per_col(df1, col_name='text_case_cover')
count_nan_per_col(df1, col_name='date_decision')


#################################
print(df.columns)
print(len(df)) # 31195

count_nan_per_col(df, col_name='decisionID')
count_nan_per_col(df, col_name='extracted_dates')
count_nan_per_col(df, col_name='LOC_HEARING')
count_nan_per_col(df, col_name='TRIBUNAL')
count_nan_per_col(df, col_name='PUBLIC_PRIVATE_HEARING')
count_nan_per_col(df, col_name='INCHAMBER_VIRTUAL_HEARING')
count_nan_per_col(df, col_name='JUDGE')
count_nan_per_col(df, col_name='text_case_cover')
count_nan_per_col(df, col_name='DATE_DECISION')
count_nan_per_col(df, col_name='decision_outcome')
