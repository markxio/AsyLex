from datasets import Dataset, load_from_disk, load_dataset
import spacy
import re 
import pandas as pd
import numpy as np
from datetime import datetime

def col_list_to_string(df, col_name=""):
    df[f'{col_name}'] = df[f'{col_name}'].astype('string')
    df[f'{col_name}'] = [s.replace('[\'', '') for s in df[f'{col_name}']]
    df[f'{col_name}'] = [s.replace('\']', '') for s in df[f'{col_name}']]
    df[f'{col_name}'] = [s.replace('\'', '') for s in df[f'{col_name}']]
    return df

def col_dict_to_string(df, col_name=""):
    df[f'{col_name}'] = [s.replace('{\'', '') for s in df[f'{col_name}']]
    df[f'{col_name}'] = [s.replace('\'}', '') for s in df[f'{col_name}']]
    year = re.compile(r'\d{4}') # flagging years

    new_dates = []
    new_time_span = []

    for index, row in df.iterrows():
        mylist = row[f'{col_name}'].split('\'') # is a string, convert to list

        try:
            mylist = list(set(mylist)) # remove duplicates
            mylist.remove(', ')
            mylist.remove('october 2007') # the date of a law, appea all the time
            mylist = [i for i in mylist if not year.match(i)] # removing single years 4digits
        except:
            pass

        if len(mylist)>0:
            max_date = max(mylist)
            min_date = min(mylist)

            try:
                min_date = datetime.strptime(min_date, "%B %d, %Y")
                max_date = datetime.strptime(max_date, "%B %d, %Y")
                new_dates.append(max_date)
                span = max_date-min_date
                new_time_span.append(span) #result is a timedelta object
            except:
                new_dates.append(max_date)
                new_time_span.append(min_date)

        else:
            new_dates.append(np.nan)
            new_time_span.append(np.nan)
        # then we consider than max date is the date of the decision
        # and we get the span of time elapsed btw min and max, which is a proxy per
        # how long to have a decision made

    max_date_formatted = []
    for date in new_dates:
        if type(date)==str:
            max_date_formatted.append(date)
        elif type(date)==datetime:
            max_date_formatted.append(date.strftime('%B %d, %Y'))

    df["decision_date"] = new_dates
    df["span_time_decision"] = new_time_span

    return df

def clean_data_case_cover(df):
    # rename first col
    df.rename(columns={"Unnamed: 0": "decisionID"}, inplace=True)

    print("###########################################")
    print("###########################################")
    print('Let\'s clean the df')
    df.drop('Unnamed: 0.1', inplace=True, axis=1)

    # convert empty cells to NaN value
    df["extracted_gpe"] = df["extracted_gpe"].apply(lambda y: np.nan if len(y) == 0 else y)
    df["extracted_org"] = df["extracted_org"].apply(lambda x: x.strip('[]')).replace('', np.nan)
    df["public_private_hearing"] = df["public_private_hearing"].apply(lambda y: np.nan if len(y) == 0 else y)
    df["in_chamber_virtual"] = df["in_chamber_virtual"].apply(lambda y: np.nan if len(y) == 0 else y)

    # convert col content to string
    col_list_to_string(df, col_name="extracted_gpe")
    col_list_to_string(df, col_name="public_private_hearing")
    col_list_to_string(df, col_name="in_chamber_virtual")
    df["in_chamber_virtual"] = df["in_chamber_virtual"].apply(lambda y: np.nan if y == "[None]" else y)
    df["public_private_hearing"] = df["public_private_hearing"].apply(lambda y: np.nan if y == "[None]" else y)

    df["extracted_gpe"] = df["extracted_gpe"].apply(lambda y: y.replace('que bec', 'quebec'))

    # we assume if it's NaN it is generally in chambers
    df["in_chamber_virtual"].replace(np.nan, "in chambers", inplace=True)

    col_dict_to_string(df, col_name='extracted_dates')

    return df

def count_nan_per_col(df, col_name =""):
    nan_count = df[f'{col_name}'].isna().sum()
    print("There are {} missing values in the column {}".format(nan_count, col_name))

def get_data_case_cover(path_case_cover_csv, path_label_outcomes_csv):
    df = pd.read_csv(path_case_cover_csv)
    df_labels = pd.read_csv(path_label_outcomes_csv, sep=";")

    # BINARY CLASSIFICATION TASK
    #df_labels = df_labels[df_labels.decision_outcome != 2]  # delete "uncertain" rows
    print("Number of case outcome annotations: ")
    print(df_labels.shape) #31226 rows
    print(df.columns)
    print("Number of case covers: ")
    print(len(df))
    
    clean_data_case_cover(df)

    # we keep the col with missing values as NaN, we just ignore it.
    df.drop(['date_decision'], axis=1,
            inplace=True)  # remove the features that seem irrelevant for now

    # TODO: do something for span time decision column
    # i.e. less than a year if one year only has been written

    print("###########################################")
    print("###########################################")
    print("Cleaning done!")
    print("Now let's explore the df a bit")
    print(df.shape)
    print(df.columns)
    count_nan_per_col(df, col_name="extracted_gpe")
    count_nan_per_col(df, col_name="extracted_org")
    count_nan_per_col(df, col_name="extracted_dates")
    count_nan_per_col(df, col_name="decision_date")
    count_nan_per_col(df, col_name="public_private_hearing")
    count_nan_per_col(df, col_name="in_chamber_virtual")
    count_nan_per_col(df, col_name="judge_name")
    count_nan_per_col(df, col_name="span_time_decision")
    count_nan_per_col(df, col_name="text_case_cover")
    

    # merge labels and features in one df, based on the index
    df_labels = df_labels.rename(columns={'Unnamed: 0': 'decisionID'}) 
    num_labelled = len(df_labels)
    df = df.drop(df[df["decisionID"] == "no_match_filename"].index)
    df['decisionID']=df['decisionID'].astype(int)
    df_labels['decisionID']=df_labels['decisionID'].astype(int)
    
    print("---------------------------------------------------")
    print(f"We have {num_labelled} silver-standard labelled cases")
    df = df.merge(df_labels, how='left', on="decisionID")
    print(df)
    
    count_nan_per_col(df, col_name="decision_outcome")  # There are 20651 missing values in the column decision_outcome
    
    df.dropna(axis=0, subset=["decision_outcome"],
              inplace=True)  # delete rows with no decision outcome value. new shape: (25233, 9)
    df['decision_outcome']=df['decision_outcome'].astype(int)
    new_shape = df.shape
    print("We delete rows with no decision outcome value found. We now have a df with shape {}".format(new_shape))

    return df


def clean_data_main_text(df):

    print("###########################################")
    print("###########################################")
    print('Let\'s clean the df')
    df = df.drop(df[df["decisionID"] == "decisionID"].index)
     
    df['CLAIMANT_EVENTS'] = df['CLAIMANT_EVENTS'].astype(str)
    df['CREDIBILITY'] = df['CREDIBILITY'].astype(str)
    df['DETERMINATION'] = df['DETERMINATION'].astype(str)
    df['CLAIMANT_INFO'] = df['CLAIMANT_INFO'].astype(str)
    df['PROCEDURE'] = df['PROCEDURE'].astype(str)
    df['DOC_EVIDENCE'] = df['DOC_EVIDENCE'].astype(str)
    df['EXPLANATION'] = df['EXPLANATION'].astype(str)
    df['LEGAL_GROUND'] = df['LEGAL_GROUND'].astype(str)
    df['LAW_CASE'] = df['LAW_CASE'].astype(str)
    df['GPE'] = df['GPE'].astype(str)
    df['DATE'] = df['DATE'].astype(str)
    df['NORP'] = df['NORP'].astype(str)
    df['ORG'] = df['ORG'].astype(str)
    df['LAW'] = df['LAW'].astype(str)
    df['LAW_REPORT'] = df['LAW_REPORT'].astype(str) 
 
    # convert empty cells to NaN value
    df['CLAIMANT_EVENTS'] = df['CLAIMANT_EVENTS'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['CREDIBILITY'] = df['CREDIBILITY'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['DETERMINATION'] = df['DETERMINATION'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['CLAIMANT_INFO'] = df['CLAIMANT_INFO'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['PROCEDURE'] = df['PROCEDURE'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['DOC_EVIDENCE'] = df['DOC_EVIDENCE'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['EXPLANATION'] = df['EXPLANATION'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['LEGAL_GROUND'] = df['LEGAL_GROUND'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['LAW_CASE'] = df['LAW_CASE'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['LAW_REPORT'] = df['LAW_REPORT'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['GPE'] = df['GPE'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['DATE'] = df['DATE'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['NORP'] = df['NORP'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['ORG'] = df['ORG'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)
    df['LAW'] = df['LAW'].apply(lambda x: x.strip('['']')).replace('\'\'', np.nan)

    count_nan_per_col(df, col_name='CLAIMANT_EVENTS')
    count_nan_per_col(df, col_name='CREDIBILITY')
    count_nan_per_col(df, col_name='DETERMINATION')
    count_nan_per_col(df, col_name='CLAIMANT_INFO')
    count_nan_per_col(df, col_name='PROCEDURE')
    count_nan_per_col(df, col_name='DOC_EVIDENCE')
    count_nan_per_col(df, col_name='EXPLANATION')
    count_nan_per_col(df, col_name='LEGAL_GROUND')
    count_nan_per_col(df, col_name='LAW_CASE')
    count_nan_per_col(df, col_name='LAW_REPORT')
    count_nan_per_col(df, col_name='GPE')
    count_nan_per_col(df, col_name='DATE')
    count_nan_per_col(df, col_name='NORP')
    count_nan_per_col(df, col_name='ORG')
    count_nan_per_col(df, col_name='LAW')

    df['CLAIMANT_EVENTS'] = df['CLAIMANT_EVENTS'].astype(str)
    df['CREDIBILITY'] = df['CREDIBILITY'].astype(str)
    df['DETERMINATION'] = df['DETERMINATION'].astype(str)
    df['CLAIMANT_INFO'] = df['CLAIMANT_INFO'].astype(str)
    df['PROCEDURE'] = df['PROCEDURE'].astype(str)
    df['DOC_EVIDENCE'] = df['DOC_EVIDENCE'].astype(str)
    df['EXPLANATION'] = df['EXPLANATION'].astype(str)
    df['LEGAL_GROUND'] = df['LEGAL_GROUND'].astype(str)
    df['LAW_CASE'] = df['LAW_CASE'].astype(str)
    df['GPE'] = df['GPE'].astype(str)
    df['DATE'] = df['DATE'].astype(str)
    df['NORP'] = df['NORP'].astype(str)
    df['ORG'] = df['ORG'].astype(str)
    df['LAW'] = df['LAW'].astype(str)
    df['LAW_REPORT'] = df['LAW_REPORT'].astype(str)

    df['CLAIMANT_EVENTS'] = df['CLAIMANT_EVENTS'].apply(lambda x: x.replace('\'', ''))
    df['CREDIBILITY'] = df['CREDIBILITY'].apply(lambda x: x.replace('\'', ''))
    df['DETERMINATION'] = df['DETERMINATION'].apply(lambda x: x.replace('\'', ''))
    df['CLAIMANT_INFO'] = df['CLAIMANT_INFO'].apply(lambda x: x.replace('\'', ''))
    df['PROCEDURE'] = df['PROCEDURE'].apply(lambda x: x.replace('\'', ''))
    df['DOC_EVIDENCE'] = df['DOC_EVIDENCE'].apply(lambda x: x.replace('\'', ''))
    df['EXPLANATION'] = df['EXPLANATION'].apply(lambda x: x.replace('\'', ''))
    df['LEGAL_GROUND'] = df['LEGAL_GROUND'].apply(lambda x: x.replace('\'', ''))
    df['LAW_CASE'] = df['LAW_CASE'].apply(lambda x: x.replace('\'', ''))
    df['LAW_REPORT'] = df['LAW_REPORT'].apply(lambda x: x.replace('\'', ''))
    df['GPE'] = df['GPE'].apply(lambda x: x.replace('\'', ''))
    df['DATE'] = df['DATE'].apply(lambda x: x.replace('\'', ''))
    df['NORP'] = df['NORP'].apply(lambda x: x.replace('\'', ''))
    df['ORG'] = df['ORG'].apply(lambda x: x.replace('\'', ''))
    df['LAW'] = df['LAW'].apply(lambda x: x.replace('\'', ''))

    return df

def get_data_main(df_main_entities, path_label_outcomes):
    df = df_main_entities
    df_labels = pd.read_csv(path_label_outcomes, sep=";")
    df = clean_data_main_text(df)
  
    # merge labels and features in one df, based on the canlii decision ID  
    df_labels = df_labels.rename(columns={'Unnamed: 0':'decisionID'})
    num_labelled = len(df_labels)
    print(f"We have {num_labelled} silver-standard labelled cases") 
    
    print(len(df))    
    #df=df.convert_dtypes()
    df.decisionID = pd.to_numeric(df.decisionID, errors='coerce').fillna(0).astype(np.int64)

    mydf = df.merge(df_labels, how='left', on="decisionID")
    print(mydf.columns)
    mydf = mydf.dropna(subset=["decision_outcome"])
    mydf.decision_outcome = pd.to_numeric(mydf.decision_outcome, errors='ignore').astype(np.int64)  
    print(mydf) 
    print("------------------------------------------------")
    count_nan_per_col(mydf, col_name="decision_outcome")

    new_shape = mydf.shape
    print("We delete rows with no decision outcome value found. We now have a df with shape {}".format(new_shape))
    print(mydf)
    return mydf

if __name__ == '__main__':

    ###################################################################################################
    ############# CREATE TRAIN SET
    path_case_cover = "case_cover_anonymised+extracted_entities.csv"
    path_label_outcomes = "data/determination_data/silver_standard_final_outcomes.csv"

    ###################################################################################################

    # get the CASE COVER data ready
    print("#############################################")
    print("#############################################")
    print("First, get the data from the case cover ready...")
    df = get_data_case_cover(path_case_cover, path_label_outcomes)

    # rename col
    df.rename(columns={'decision_date': "DATE_DECISION", "extracted_gpe": "LOC_HEARING",
                       "extracted_org": "TRIBUNAL", 'public_private_hearing': "PUBLIC_PRIVATE_HEARING",
                       "in_chamber_virtual": "INCHAMBER_VIRTUAL_HEARING", "judge_name": "JUDGE"}, inplace=True)
    df.drop("span_time_decision", axis=1, inplace=True)

    # convert all col types
    df["DATE_DECISION"] = df['DATE_DECISION'].astype(str)
    df['LOC_HEARING'] = df['LOC_HEARING'].astype(str)
    df['TRIBUNAL'] = df['TRIBUNAL'].astype(str)
    df['PUBLIC_PRIVATE_HEARING'] = df['PUBLIC_PRIVATE_HEARING'].astype(str)
    df['INCHAMBER_VIRTUAL_HEARING'] = df['INCHAMBER_VIRTUAL_HEARING'].astype(str)
    df['JUDGE'] = df['JUDGE'].astype(str)

    df.reset_index(inplace=True)
    df.rename(columns={'CaseID': "decisionID"}, inplace=True)

    print(df.head(10))
    print(len(df))
    df.to_csv("case_cover_entities_decision_outcome_9june.csv", sep=";")
    print("#############################################")
    print("#############################################")
    ###################################################################################################

    

    ###################################################################################################
    print("Second, get the data from the main text ready...")
    dataset_maintext = load_from_disk("all_inference_dataset")
    dataset_maintext = dataset_maintext.remove_columns(['__index_level_0__', '__index_level_1__','__index_level_2__'])
    print(dataset_maintext)
    df_maintext = dataset_maintext.to_pandas()
    
    print(len(df_maintext))

    df_main = get_data_main(df_maintext, path_label_outcomes)

    # get it in pandas to merge them on the same col
    # merge two dfs based on common key column (here canlii ID)
    # merge deletes row for which an index has not been found in both dataframes
    print("###########################################")
    print("Let's merge the 2 dataframes case cover and main to get labelled data")

    result_df = pd.merge(df_main, df, how='outer')
    result_df.drop(['index'], axis=1, inplace=True)
    print(result_df.columns)
    print(result_df)
    print(result_df.index)
    result_df.to_csv("main_and_case_cover_12june.csv", sep=';')
    print("..done")
    exit()

    print("###########################################")
    print("###########################################")
    ###################################################################################################
    ############# CREATE TEST SET
    exit()

    ############ get the df ready: we concatenate all col to form one col of text per case
    # we put a separation token btw each col: "[SEP]"
    df["text"] = df[['loc_hearing', 'extracted_org', 'public_private_hearing', 'in_chamber_virtual', 'judge_name', 'decision_date']].apply("[SEP]".join, axis=1)

    # now we can drop all useless col and only keep text, decisionID and outcome
    df.drop(['loc_hearing', 'tribunal', 'public_private_hearing', 'in_chamber_virtual', 'judge_name', 'decision_date'], axis=1, inplace=True)
    df = df[["text", "decision_outcome"]]
    df.rename({"decision_outcome": "labels"}, axis=1, inplace=True)
    df["labels"] = df["labels"].astype(int)
    print(df.dtypes) # the type of each column
    print("Done!")


