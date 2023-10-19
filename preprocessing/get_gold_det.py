import pandas as pd
import re

f = "gpu_silver_standard_decision_outcome.csv"
df = pd.read_csv(f, usecols=["decisionID", "extracted_sentences_determination"], sep=";")
print(len(df))
df = df[df.decisionID != 'no_match_filename']
print(len(df))

print("then we load our gold standard annotations")
# path to the CSV woth abt 2,000 labelled decisions
my_csv_labels = "gold_outcome_cases.csv"
# this df contains decision for 1400 cases, as 0 = reject, 1 = accept
df_labels = pd.read_csv(my_csv_labels, sep=";", index_col=None, usecols=["decisionID", "decision"])
print("Number of gold standard annotated cases: ")
print(len(df_labels))
df_labels["decisionID"] = df_labels["decisionID"].astype("Int64")
df["decisionID"] = df["decisionID"].astype("float")
df["decisionID"] = df["decisionID"].astype("Int64")

df_labels["decisionID"] = df_labels["decisionID"].round(0)
print(df_labels.head())

# get the cleaned decision Id and not the file path
def change_col_get_id(df):
    list_id = []
    for row in df.iterrows():
        num_id = re.compile('(?<=canlii)\d{1,6}(?=\.pdf)')
        decision_id = num_id.search(str(row))
        if decision_id:
            decision_id = decision_id.group(0) # gets only the decision ID
        list_id.append(decision_id)

    return list_id

#list_id = change_col_get_id(df_labels)
#df_labels["file_path"] = list_id

print("Number of gold standard annotations: ")
print(len(df_labels)) # 1400
#df_labels.rename({'file_path': 'decisionID'}, axis=1, inplace=True) # rename col
print("we print the gold standard ds as a check:")

print(df["decisionID"].dtypes)
print(df_labels["decisionID"].dtypes)

# set the decision ID col as index
df.set_index('decisionID', inplace=True)
df_labels.set_index('decisionID', inplace=True)
print(df_labels.head())

# merge two dfs based on common key column (here canlii ID)
# merge deletes row for which an index has not been found in both dataframes
print("###########################################")
print("Let's merge the 2 dataframes to get labelled data")
result_df = pd.merge(df, df_labels, how='inner', left_index=True, right_index=True)

# concat keeps all rows and fills empty ones with NaN or 0 if attaching: .fillna(0)
# result_df = pd.concat([mydf, df_judges_date], axis=1)
print("merged df...")
print(result_df.head(10))
print(len(result_df))
print("merged df ends.. result_df is the created df!")

print("###########################################")
print("Writing the df to a .CSV file")

# write the csv -- this overwrites and writes a whole new csv file
result_df.to_csv("gold_cases_outcome_sentences.csv", encoding='utf-8', sep=";", index=True, header=True)

