# This script gets the necessary training data for the sentence classifier
# ie it outputs a csv with case ID and the sentences flagged by the NER with the determination label


import pandas as pd
import spacy
import re
from pathlib import Path
import tqdm
import numpy as np


def clean_raw_data(df):
    df.dropna(axis=0, inplace=True)
    return df


print("###########################################")
print("Now we apply the NER model_HF_distilbert_sentences_clf to get extracted sentences flagged as Determination sentences")
######### apply NER
def process_by_row (df):
    """
    :param df: input is a df with index is the decision id and second col is the raw text
    loops through every row of the df, ie it processed one decision at a time
    :return: updates the dataframe with the column with the NER results
    """
    outcome_to_df = []

    for idx, row in tqdm.tqdm(df.iterrows()):

        # row_ner is the full pre-processed text of the first page of a decision
        row_ner = list(ner.pipe(row))

        # applying NER pipeline to row_ner, ie to one decision at a time
        # my_ner_output =[] # this is to use if you want all extracted items in one col
        outcome = []

        for item in row_ner:

            for ent in item.ents:
                #output = [ent.text, ent.start_char, ent.end_char, ent.label_]

                if ent.label_=="DETERMINATION":
                    output_date = ent.text
                    outcome.append(output_date)


        # my_ner_output: this returns a list of list, containing all entities for one decision
        outcome_to_df.append(outcome) # is a list of lists of lists (oops)

    # write my_ner_output to dataframe
    df["extracted_sentences_determination"] = outcome_to_df
    #print(outcome_to_df)
    return df



if __name__ == '__main__':
    print("###########################################")
    print("We start by getting the data from a .CSV file and clean it: the last 10 sentences of each case")
    raw_sentences_csv = "./data/determinations_sentences/last_10p_sentences+decisionID.csv"

    df = pd.read_csv(raw_sentences_csv, sep=';', index_col=None)
    old_len=len(df)
    df = df.iloc[:, 1:]  # dropping first column

# Clean the dataframe, delete empty rows and first col


    df=clean_raw_data(df)
    print(df.head())
    print(df.shape) # (1164786, 2)
    new_len = len(df)
    print("We cleaned the df from empty rows and decreased the size from {} to {}, ie we deleted {}".format(old_len, new_len, (old_len-new_len)))
    uniques = df["decisionID"].unique()
    print(f"We have determination sentences for {len(uniques)} cases")
    print("###########################################")


# Load trained NER pipeline package
    ner = spacy.load("./model_legalbert_scratch/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0")
    print("###########################################")
    print("Now we apply the NER: inference to get extracted sentences flagged as Determination sentences")
######### apply NER
    print("###########################################")

    process_by_row(df)


# remove rows where no sentences was extracted by the NER
    df["extracted_sentences_determination"] = df["extracted_sentences_determination"].apply(lambda y: np.nan if len(y)==0 else y) # convert empty list to NaN value
    print("With the NER based on LegalBERT, we flagged a total determination sentences of: ")
    df=clean_raw_data(df) # clean from empty rows
    df.drop('Text', axis=1, inplace=True)
    print("NER done!, here the result in the dataframe")
    print(len(df))

    df.to_csv("silver_standard_decision_outcome.csv", sep=";")



"""
print("then we load our gold standard annotations")
# path to the CSV woth abt 2,000 labelled decisions
my_csv_labels = "/data/determination_sentences/gold_outcome_cases.csv"
# this df contains decision for 1400 cases, as 0 = reject, 1 = accept
df_labels = pd.read_csv(my_csv_labels, sep=";", index_col=None)

# get the cleaned decision Id and not the file path
def change_col_get_id(df):
    list_id = []
    for row in tqdm.tqdm(df.iterrows()):
        num_id = re.compile('(?<=canlii)\d{1,6}(?=\.pdf)')
        decision_id = num_id.search(str(row))
        if decision_id:
            decision_id = decision_id.group(0) # gets only the decision ID
        list_id.append(decision_id)

    return list_id

list_id = change_col_get_id(df_labels)
df_labels["file_path"] = list_id

print("Number of gold standard annotations: ")
print(len(df_labels)) # 1400
df_labels.rename({'file_path': 'decisionID'}, axis=1, inplace=True) # rename col
print("we print the gold standard ds as a check:")

# set the decision ID col as index
df.set_index('decisionID', inplace=True)
df_labels.set_index('decisionID', inplace=True)
print(df_labels.head())


# merge two dfs based on common key column (here canlii ID)
# merge deletes row for which an index has not been found in both dataframes
print("###########################################")
print("Let's merge the 2 dataframes to get labelled data")
result_df = pd.merge(df, df_labels, how='left', left_index=True, right_index=True)

# concat keeps all rows and fills empty ones with NaN or 0 if attaching: .fillna(0)
# result_df = pd.concat([mydf, df_judges_date], axis=1)
print("merged df...")
print(result_df.head(10))
print(len(result_df))
print("merged df ends.. result_df is the created df!")

print("###########################################")
print("Writing the df to a .CSV file")

# write the csv -- this overwrites and writes a whole new csv file
result_df.to_csv("sentences+decision_label.csv", encoding='utf-8', sep=";", index=True, header=True)

"""
