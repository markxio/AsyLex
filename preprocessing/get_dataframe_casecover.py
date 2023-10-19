# This script outputs a dataframe from the information extracted (in other repo: Refugee_cases_analysis)
# It outputs a dataframe, one row per decision
# Data is structured by category
# it contains the decision outcome
# this is the putting together my training data, then i can test model_HF_distilbert_sentences_clf on this

import pandas as pd
import glob
import re
from pathlib import Path
import PyPDF2
import spacy
import tqdm
from fuzzywuzzy import fuzz


def get_pdf_handle(file_path):
    return PyPDF2.PdfFileReader(file_path)

# initialize for basic text cleaning
nls = re.compile(r'(?<![\r\n])(\r?\n|\n?\r)(?![\r\n])')
spaces = re.compile(r'\s+')

def clean_raw_text(raw_str):
    """
    :param raw_str: a string from one single decision
    :return: the string cleaned, containing only the text we want to keep in the df
    """

    # lowercase every strings
    raw_str = raw_str.lower()

    # Remove multiple consecutive spaces
    clean = spaces.sub(' ', raw_str)

    # Remove random newlines
    clean = nls.sub(' ', clean)

    return clean

#faulty_empty_pdfs = 0
def get_pdf_first_page_text(pdf_file):
    try:
        mypdf = get_pdf_handle(pdf_file) #read the PDFs
        #numpages = mypdf.getNumPages()
        first_page = mypdf.getPage(0)
        return first_page.extractText()
    except:
        pass

def create_data_to_dict(base_path_root):
    """
    :param base_path_root: base path of the PDF files folders
    :return: a dict, key is the number of a decision, value is the raw text of the first page
    """

    data = {}

    # analysis/run statistics
    n_files = 0

    # gets path for pdf
    source_folder_pdf = Path(base_path_root).expanduser()

    # get all files in the folder as a list of paths
    files_list_pdf = glob.glob(f"{source_folder_pdf}/*.pdf")
    print(f"-- Folder {base_path_root} holds {len(files_list_pdf)} .pdf files")

    # loop: per decision file
    for decision_file_path in tqdm.tqdm(files_list_pdf[:100]):
        try:
            first_page = get_pdf_first_page_text(decision_file_path)

            # text cleaning
            first_page = clean_raw_text(first_page)

            n_files += 1

            # 2002canlii52643.pdf -- get the decision Id as key to the created dict
            with open(decision_file_path, "r", encoding='utf8') as fileid:
                decision_id = re.search('(?<=canlii)\d{1,6}(?=\.pdf)', decision_file_path)
                if decision_id:
                    decision_id = decision_id[0]
                else:
                    decision_id = "no_match_filename"
                # add new item to dictionary per decision, iteratively
                # key is the decisionID, value is the text string of first page
                data[decision_id] = str(first_page)


        except:
            pass

    print(f"Files processed: %d" % n_files)
    print(f"----------------------------------------------------")
    return data

# load pipeline
# TODO: may be worth changing to CNN+rsv model_HF_distilbert_sentences_clf instead.
ner = spacy.load(Path("NER//PACKAGES/en_cover_baseline-0.0.0/en_cover_baseline/en_cover_baseline-0.0.0"))

def process_by_row (df):
    """
    :param df: input is a df with index is the decision id and second col is the raw text
    loops through every row of the df, ie it processed one decision at a time
    :return: updates the dataframe with the column with the NER results
    """
    dates_to_df = []
    gpe_to_df = []
    org_to_df = []
    person_to_df = []
    public_to_df = []
    virtual_to_df = []

    for idx, row in tqdm.tqdm(df.iterrows()):

        # row_ner is the full pre-processed text of the first page of a decision
        row_ner = list(ner.pipe(row))

        # applying NER pipeline to row_ner, ie to one decision at a time
        # my_ner_output =[] # this is to use if you want all extracted items in one col
        dates = []
        gpe = []
        org = []
        person = []
        public = []
        virtual = []

        for item in row_ner:

            ## the last 2 columns are rule-based extraction / not from the NER
            # from regex search
            private_public_pattern = re.compile("/(private\s*?proceedings?)|(private\s*?hearrings?)|(procédure\s*publique)|(audience\s*publique)|(public\s*?proceedings?)|(public\s*?hearings?)|(huis\s*clos)/gi")
            private_public = private_public_pattern.search(str(item))
            if private_public:
                private_public = private_public.group(0)
            public.append(private_public)

            virtual_pattern = re.compile('/(virtual)|(in\s*?chambers?)|(video-?conference)|(in\s*?camera?)|(in\s*?person?)|(telephone)/gi')
            virtual_inperson = virtual_pattern.search(str(item))
            if virtual_inperson:
                virtual_inperson = virtual_inperson.group(0)
            virtual.append(virtual_inperson)

            for ent in item.ents:
                #output = [ent.text, ent.start_char, ent.end_char, ent.label_]

                if ent.label_=="DATE":
                    output_date = ent.text
                    dates.append(output_date)

                if ent.label_=="GPE":
                    output_gpe = ent.text
                    gpe.append(output_gpe)

                if ent.label_=="ORG":
                    output_org = ent.text
                    org.append(output_org)

                if ent.label_ == "PERSON":
                    output_person = ent.text
                    person.append(output_person)

                #output = [ent.text, ent.label_]
                #my_ner_output.append(output)

        # my_ner_output: this returns a list of list, containing all entities for one decision
        dates_to_df.append(dates) # is a list of lists of lists (oops)
        gpe_to_df.append(gpe)
        org_to_df.append(org)
        person_to_df.append(person)
        public_to_df.append(public)
        virtual_to_df.append(virtual)

    # write my_ner_output to dataframe
    df["extracted_dates"] = dates_to_df
    df["extracted_gpe"] = gpe_to_df
    df["extracted_org"] = org_to_df
    df["extracted_persons"] = person_to_df
    df["public_private_hearing"] = public_to_df
    df["in_chamber_virtual"] = virtual_to_df

    #print(df.head())
    #print(df.tail())
    return df, dates_to_df, gpe_to_df, org_to_df, person_to_df, public_to_df, virtual_to_df

def ratio_levenshtein_similarity(str1, str2):
    #fuzz uses Levenshtein Distance
    # will return 100 if there is a word match
    similarity_ratio = fuzz.token_set_ratio(str1, str2)
    # print(similarity_ratio)
    return similarity_ratio

def get_judge_col_cross_comparison(df):
    """
    input: the 2 columns of the dataframe
    we compare the 2 col and find the common name. This name in common is the name of the judge
    :return: a list of judge names
    """
    list_judges = []

    for extracted_person, ner_output in zip(result_df["extracted_persons"], result_df["NER_output"]):
        # print(type(extracted_person), type(ner_output)) # extracted person is a list, ner_output a str
        ner_output = str(ner_output) # there is one type integer
        ner_output_as_list = re.sub("'", "", ner_output)
        ner_output_as_list = ner_output_as_list.strip('][').split(', ')
            # ner_output_as_list = ner_output_as_list.strip('][').split(' ')
            # print(ner_output_as_list)
            # converting my string-type list to list
            # print(type(extracted_person), type(ner_output_as_list))
            # ner_output_as_list = ast.literal_eval(ner_output)
        matches = set(extracted_person).intersection(ner_output_as_list)  # matches is a set, not a list (can convert set to list)
            # print(f"extracted: {extracted_person}, ner: {ner_output}, matches: {matches}")

        if matches == set():
            for item in ner_output_as_list:
                similarity_ratio = ratio_levenshtein_similarity(extracted_person, ner_output)
                if similarity_ratio == 100:
                    matches = item
                        # print(matches)

        list_judges.append(matches)

        # print(list_judges)
    return list_judges

def get_decision_data_col_cross_comparison(df):
    """
    input: the 2 columns of the dataframe
    we compare the 2 col and find the common date. This date in common is the date of the decision
    :return: a list of dates
    """
    list_dates = []

    for extracted_dates, ner_output in zip(result_df["extracted_dates"], result_df["NER_output"]):
        # print(type(extracted_dates), type(ner_output)) # <class 'list'> <class 'str'>
        ner_output = str(ner_output)
        ner_output = re.sub("(?<=\d),", '', ner_output) # delete the comma in dates
        ner_output_as_list = re.sub(r"'", "", ner_output)
        ner_output_as_list = ner_output_as_list.strip('][').split(', ')
        # print(ner_output_as_list)
        # print(extracted_dates)
        matches = set(extracted_dates).intersection(ner_output_as_list)

        if matches == set():
            for item1 in ner_output_as_list:
                    for item2 in extracted_dates:
                        similarity_ratio = ratio_levenshtein_similarity(extracted_dates, ner_output_as_list)
                        # print(similarity_ratio)
                        if similarity_ratio > 90: #85 here because of the comma, may not be 100.
                            matches = item1
                            # print(matches)

        list_dates.append(matches)


    return list_dates

def remove_duplicates_from_list(mylist):
    # print(mylist)
    new_list = set(mylist)
    return new_list

if __name__ == '__main__':

    # here we have the base path, later we add the trailing digits in a for loop
    base_path_root_all_pdfs = "refugee_cases_analysis/pdf_tar/PDF-inputfile-cases-collected_2022-10-24-13-26-00.csv-nowis2022-10-28-16-45-40"
    # base_path_root_all_pdfs = "refugee_cases_analysis/pdf_tar/test_sample"

    data_dict = create_data_to_dict(base_path_root_all_pdfs) # creates a dict, key is the decision ID and value is a string of text of the first page

    # create a pandas dataframe from a python dictionary
    mydf = pd.DataFrame.from_dict(data_dict, orient='index') # decisionID is the index, i.e. the first column
    mydf.columns = ["text_case_cover"]

    # clean the dataframe by removing rows which contain "False" for the 1st page
    mydf = mydf[mydf.iloc[:, 0] != 'False']

    #############################################################################################

    # creates a CSV file from the df
    # use a seperator that is not used in the text of the first page
    # first column is the case number, second column is the raw text of the first page
    # mydf.to_csv("first_page_all_cases.csv", encoding='utf-8', sep=";", index=True, header=True)


    #############################################################################################
    # Use this if you want to have all entities in one, not structured by decision
    # transforming the df to a list of strings
    # this is the format of input data_first_page required by spacy to operate their model_HF_distilbert_sentences_clf
    # (the .tolist would create a list of lists instead which is why we dont use it)

    #mydata = process_all_cells(mydf)
    #############################################################################################


    ########################### Deploy the NER model_HF_distilbert_sentences_clf to the whole dataset  #############################

    # searching for entities at the document level, ie per each decision as one string of text
    # we add a column to the initial dataframe mydf and we add it to the CSV file
    # we add a header corresponding to the architecture used / model_HF_distilbert_sentences_clf used

    process_by_row(mydf) # updates the dataframe, adding columns
        # Index(['text_case_cover', 'extracted dates', 'extracted gpe', 'extracted org',
        #        'extracted persons', 'public/private hearing', 'in chamber/virtual'],
        #       dtype='object')


    ######  get the name of the judge
    print("###########################################")
    print("###########################################")
    print("Getting the name of the judge...")
    # from the .csv created in "get_judge_name.py" script
    # the decision ID is the index of the df (first col)
    df_judges_date = pd.read_csv("./data/Judges_names.csv", index_col=0)
    df_judges_date.columns = ['NER_output'] # naming column for easier access
    # print(df_judges_date.shape) # (45942, 1)

    #print("mydf.index.values...")
    mydf.sort_index(inplace=True)
    try:
        mydf.index = mydf.index.astype('int64', copy=False)
    except:
        pass

    print("Done! Now adding the column to the dataset")

    # print(df_judges_date.head())
    # print(mydf.head(10))

    # merge two dfs based on common key column (here canlii ID)
    # merge deletes row for which an index has not been found in the df on the left (here mydf)
    result_df = pd.merge(mydf, df_judges_date, how='left', left_index=True, right_index=True).fillna(0)

    # concat keeps all rows and fills empty ones with NaN or 0 if attaching: .fillna(0)
    # result_df = pd.concat([mydf, df_judges_date], axis=1)
    print("merged df...")
    #print(result_df.head(10))
    #print(result_df.columns)
    print("merged df ends.. result_df is the created df!")
    print("the dataframe has {} rows".format(len(result_df)))
    print("###########################################")

    print("Now comparing names across columns to keep only the name of the judge")
    list_judges = get_judge_col_cross_comparison(result_df)
    result_df["judge_name"] = list_judges
    # print(result_df["judge_name"].head(10))

    # cleaning of the new created column, converting the sets to strings
    for i in range(len(result_df["judge_name"])):
        result_df["judge_name"][i] =''.join(result_df["judge_name"].iloc[i])
    # print(result_df.head(10))
    # print(type(result_df["judge_name"].iloc[0]))

    # remove duplicates in dates
    for i in range(len(result_df["extracted_dates"])):
        extracted_dates = list(result_df["extracted_dates"].iloc[i])
        extracted_dates = remove_duplicates_from_list(extracted_dates)
        result_df["extracted_dates"].iloc[i] = extracted_dates

    print("###########################################")
    print("###########################################")
    print("Now we get the date for the decision")
    # get date of the decision
    list_date_decision = get_decision_data_col_cross_comparison(result_df)
    # print(list_date_decision)
    result_df["date_decision"] = list_date_decision

    ## cleaning of the new created column, converting the sets to strings
    for i in range(len(result_df["date_decision"])):
        result_df["date_decision"].iloc[i] = ''.join(result_df["date_decision"].iloc[i])

    print("Done!")
    print(result_df.head(10))
    print("###########################################")
    print("###########################################")
    print("We write the result to a CSV file")


    # write the csv -- this overwrites and writes a whole new csv file
    result_df.to_csv("Case_cover_features.csv", encoding='utf-8', sep=";", index=True, header=True)

    # update the existing csv with a new column, not overwriting the columns stored before, just adds a new col
    # newdf = pd.read_csv('first_page_all_cases.csv', encoding='utf-8', sep=";")
    # newdf["model_prodigy_update_50annotations"] = mydata
    # newdf.to_csv('first_page_all_cases.csv', encoding='utf-8', sep=";", index=True, header=True)
