# name of the judge is at the end of a case
# after the last paragraph
# sometimes formated as a table in html
# sometimes with the tag "member"
# after is usually the date of the decision

# search for the p tag that contains member and in the same key tag search for the spans in Beautifulsou[
# also written in bold most of the time
# find parent

from bs4 import BeautifulSoup
import re
import glob
from pathlib import Path
import spacy
import pandas as pd
import tqdm

def html_directory_to_dict(base_path_root_html_folder):
    # this function creates a dict
    # key is the decision id
    # value is a list of paragraphs (for one single decision)

    # gets path for html files
    source_folder_html = Path(base_path_root_html_folder).expanduser()

    # get all files in the folder as a list of paths
    files_list_html = glob.glob(f"{source_folder_html}/*.html")
    print(f"-- Folder {base_path_root_html_folder} holds {len(files_list_html)} .html files")


    # creating the dict, key will be the decision id, then each value is a paragraph
    dict_main_text = {}

    # analysis/run statistics
    n_files = 0

    print(f"Files processed: %d" % n_files)
    return dict_main_text

def get_judge_name_from_table(my_case_path):
    # HTML parser, using BeautifulSoup
    # this function will work if the name of the judge/date is in a TABLE only
    f = open(my_case_path, 'r', encoding="utf8")
    soup = BeautifulSoup(f, 'html.parser')
    f.close()
    find_tables = soup.find_all('table') # find all tables in the HTML
    tables = list(find_tables) # convert to list

    # initialize
    clean_list = []
    list_ner = []  # initialize final list of phrases extracted

    # sometimes there are no tables at all in the case
    if len(find_tables) > 0:

        # TODO: get the last two tables maybe for when there are 2 panel members
        # or do a double check
        # but may end up having also the table from the first page
        # TODO: find a way to improve robustness of extraction here
        table = find_tables[-1] # get only the last table of the HTML
        text_table = table.get_text()
        text_table = text_table.lower() #lower case
        text = text_table.split("\n") # is a list
        text = list(filter(None, text))
        text = [x for x in text if x !='\xa0']

        for string in text:
            spaces = re.compile(r'^\s+') # regex to find spaces
            string = spaces.sub('', string) # remove spaces at the beginning of a string only
            clean_list.append(string)

        #print(clean_list)

        # look for tokens that are indicative of the wrong table being retrieved
        regex_for_triage = re.compile("appell*ant|repondant|intim|19[^9]\d|appeal|ottawa|toronto")

        # load NER pipeline
        # TODO: use the one that works the best
        ner = spacy.load(Path("./PACKAGES/en_cover_baseline-0.0.0/en_cover_baseline/en_cover_baseline-0.0.0"))

        for item in clean_list:
            if regex_for_triage.search(item):
                clean_list.clear()
                list_ner = ["no table found"]
            else:
                doc = ner(item)
                for ent in doc.ents:
                    if ent.label_ == 'DATE' or 'PERSON':
                        list_ner.append(doc.ents)

    # print(list_ner)
    return list_ner

def get_judge_name_member(my_case_path):
    f = open(my_case_path, 'r', encoding="utf8")
    soup = BeautifulSoup(f, 'html.parser')
    f.close()

    parent = soup.find_parent(name='a', attrs={'name':'member'}) # returns a Bs object


#<table class="MsoNormalTable"
#< p class ="MsoHeader"
#< b > < span XXXX Edward Bosveld  </b></p>
#May 23, 2017 </b></p>

def convertTuple(tup):
    text = ''.join(map(str, tup))
    return text

if __name__ == '__main__':
    my_test_case_path = "refugee_cases_analysis/tar_html/HTML-inputfile-cases-collected_2022-10-24-13-26-00.csv-nowis2022-10-24-15-52-54"

    # gets path for html files
    source_folder_html = Path(my_test_case_path).expanduser()

    # get all files in the folder as a list of paths
    files_list_html = glob.glob(f"{source_folder_html}/*.html")
    print(f"-- Folder {my_test_case_path} holds {len(files_list_html)} .html files")

    processed_files = 0

    # creating one dict, key will be the decision id, then each value is a list of extracted items
    dict_extract = {}

    # show the progress by iterating over the list of files
    for file in tqdm.tqdm(files_list_html):
        processed_files += 1

        my_list = get_judge_name_from_table(file) # this returns a list, for one case - is a list if tuples

        my_list_cleaned = []
        # cleaning: removing parenthesis
        for item in my_list:
            item = convertTuple(item)
            my_list_cleaned.append(item)

        # print(my_list_cleaned)

        # getting the key for the dictionary ie the decision id
        # 2004canlii69525.html -- get the decision Id as key to the created dict
        with open(file, "r", encoding='utf8') as fileid:
            decision_id = re.search('(?<=canlii)\d{1,6}(?=\.html)', file)
            if decision_id:
                decision_id = decision_id[0]
            else:
                decision_id = "no_match_filename"

        # create a list per decision, that is stored as value in the dict
        # add one entry to the list per paragraph in decision
        # key is the decisionID, value is list of extracted names and dates
        dict_extract[decision_id] = []

        # with flattened list of lists to a single list
        dict_extract[decision_id].append(my_list_cleaned)

    #print(dict_extract)

    print(f"----------------------------------------------------")
    print(f"Files processed: %d" % processed_files)

    mydf = pd.DataFrame.from_dict(dict_extract, orient='index')

    #clean from rows that don't contain names
    # mydf = mydf[mydf.iloc[:, 0] != '[]']
    # mydf = mydf[mydf.iloc[:, 0] != ['no table found']]

    print(mydf.head())
    print(mydf.tail())
    mydf.to_csv('Judges_names.csv')







