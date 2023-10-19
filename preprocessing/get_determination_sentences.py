from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import spacy
import re
from pathlib import Path
import tqdm
import numpy as np


def clean_raw_data(df):
    df.dropna(axis=0, inplace=True)
    return df


def process_by_row (chunk_df, core_id, n_cores):
    """
    :param df: input is a df with index is the decision id and second col is the raw text
    loops through every row of the df, ie it processed one decision at a time
    :return: updates the dataframe with the column with the NER results
    """
    outcome_to_df = []

    print("---------------------------------------")
    print(f"Processing {core_id}/{n_cores}...")
    df = chunk_df
    
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
    
    # remove rows where no sentences was extracted by the NER
    df["extracted_sentences_determination"] = df["extracted_sentences_determination"].apply(lambda y: np.nan if len(y)==0 else y) # convert empty list to NaN value
    print("With the NER based on LegalBERT, we flagged a total determination sentences of: ")
    print(len(df))
    df=clean_raw_data(df) # clean from empty rows
    df.drop('Text', axis=1, inplace=True)
    print("NER done!, here the result in the dataframe")
    df.to_csv(f"determination_sent_3June_{i}.csv", sep=";")
    print("printed to output file!. The END.")



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
    ner = spacy.load("../model_legalbert_scratch/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0")
    print("###########################################")
    print("Now we apply the NER: inference to get extracted sentences flagged as Determination sentences")
######### apply NER
    print("###########################################")
    n_cores = multiprocessing.cpu_count()
    print(f"running in parallel on {n_cores} cores...")
    
    chunks_df = np.array_split(df, n_cores)
    print(chunks_df)
    Parallel(n_jobs=n_cores)(delayed(process_by_row)(chunks_df[i], i, n_cores) for i in range(0, n_cores))


    # process_by_row(df)
