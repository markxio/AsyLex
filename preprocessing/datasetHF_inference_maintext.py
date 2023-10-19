# This script outputs a dataset (hugginface) from the information extracted from the labels
# We use the LegalBERT implementation of the NER since it gives the best results on average
# It goes through each row of text (a sentences per row) and gets our target entities
# Data is structured by category. one target category per column
# The input df is the text of the decision split by paragraph: all_sentences_anonymised.csv

from pathlib import Path
import spacy
import tqdm
import psutil # this is to check RAM memory usage
import time
from datasets import load_dataset
import multiprocessing
import pandas as pd
from pprint import pprint

def inference_ner(batch):
    # non-batch vs batch_size=2:
    # {'decisionID': '134491', 'Text': 'le ministre de la sécurité publique et de la   protection civile'}
    # {'decisionID': ['91207', '89936'], 'Text': ['iad file no. / node dossier de la sai : vb5-02246', 'minister et de l’immigration']}
    # print(batch)

    # we need an empty list for batch[GPE] to be able to write to batch[GPE][i]
    batch["GPE"] = []
    batch["DATE"] = []
    batch["NORP"] = []
    batch["ORG"] = []
    batch["LAW"] = []

    #for i in range(len(batch["decisionID"])):  # range(batch_size)
        # batch["Text"] is a list of paragraphs, len(list) = batch_size
    for doc in list(NER_PRETRAINED.pipe(batch["Text"], batch_size=250)):
        # using as many processes as cpu s available
            # pipe iterates through the list + batch processing

        #doc = NER_PRETRAINED(str(batch["Text"][i]))

        # decisionID = str(batch['decisionID'])
        # batch['caseID'] = decisionID
        # batch[target_ent] = [ent.text for ent in doc.ents if ent.label_ == target_ent] or [""]
        tmp = [ent.text for ent in doc.ents if ent.label_ == "GPE"] or [""]
        batch["GPE"].append(tmp)

        tmp1 = [ent.text for ent in doc.ents if ent.label_ == "DATE"] or [""]
        batch["DATE"].append(tmp1)

        tmp2 = [ent.text for ent in doc.ents if ent.label_ == "NORP"] or [""]
        batch["NORP"].append(tmp2)

        tmp3 = [ent.text for ent in doc.ents if ent.label_ == "ORG"] or [""]
        batch["ORG"].append(tmp3)

        tmp4 = [ent.text for ent in doc.ents if ent.label_ == "LAW"] or [""]
        batch["LAW"].append(tmp4)

    batch["CLAIMANT_EVENTS"] = []
    batch["CREDIBILITY"] = []
    batch["DETERMINATION"] = []
    batch["CLAIMANT_INFO"] = []
    batch["PROCEDURE"] = []
    batch["DOC_EVIDENCE"] = []
    batch["EXPLANATION"] = []
    batch["LEGAL_GROUND"] = []
    batch["LAW_CASE"] = []
    batch["LAW_REPORT"] = []

    #for i in range(len(batch["decisionID"])):  # range(batch_size)
    for doc in list(NER_SCRATCH.pipe(batch["Text"], batch_size=250)):
        #doc = NER_SCRATCH(str(batch["Text"][i]))

        tmp5 = [ent.text for ent in doc.ents if ent.label_ == "CLAIMANT_EVENTS"] or [""]
        batch["CLAIMANT_EVENTS"].append(tmp5)

        tmp6 = [ent.text for ent in doc.ents if ent.label_ == "CREDIBILITY"] or [""]
        batch["CREDIBILITY"].append(tmp6)

        tmp7 = [ent.text for ent in doc.ents if ent.label_ == "DETERMINATION"] or [""]
        batch["DETERMINATION"].append(tmp7)

        tmp8 = [ent.text for ent in doc.ents if ent.label_ == "CLAIMANT_INFO"] or [""]
        batch["CLAIMANT_INFO"].append(tmp8)

        tmp9 = [ent.text for ent in doc.ents if ent.label_ == "PROCEDURE"] or [""]
        batch["PROCEDURE"].append(tmp9)

        tmp14 = [ent.text for ent in doc.ents if ent.label_ == "DOC_EVIDENCE"] or [""]
        batch["DOC_EVIDENCE"].append(tmp14)

        tmp10 = [ent.text for ent in doc.ents if ent.label_ == "EXPLANATION"] or [""]
        batch["EXPLANATION"].append(tmp10)

        tmp11 = [ent.text for ent in doc.ents if ent.label_ == "LEGAL_GROUND"] or [""]
        batch["LEGAL_GROUND"].append(tmp11)

        tmp12 = [ent.text for ent in doc.ents if ent.label_ == "LAW_CASE"] or [""]
        batch["LAW_CASE"].append(tmp12)

        tmp13 = [ent.text for ent in doc.ents if ent.label_ == "LAW_REPORT"] or [""]
        batch["LAW_REPORT"].append(tmp13)

    return batch

    def get_ent_pretrained(example, target_ent=""):
        doc = NER_PRETRAINED(str(example["Text"]))  # a spacy doc object
        example[target_ent] = [ent.text for ent in doc.ents if ent.label_ == target_ent] or [""]
        return example[target_ent]

    def get_ent_scratch(example, target_ent=""):
        doc = NER_SCRATCH(str(example["Text"]))  # a spacy doc object
        example[target_ent] = [ent.text for ent in doc.ents if ent.label_ == target_ent] or [""]
        return example[target_ent]


if __name__ == '__main__':

    # Parallelizing processes on the CPU
    #n_cores i= multiprocessing.cpu_count()
    #print(f"Number CPU used: {n_cores}")

    #######################    #######################    #######################
    print("####################################################################################")
    print("let's get the data file and turn it into a hugging face dataset object")
    # to get the data from .csv and turn it into tar file that we then put into HF dataset

    # HF: Datasets also supports loading datasets from Pandas DataFrames
    # with the from_pandas() method
    #dataset = Dataset.from_pandas(input_df)

    data_file_path = "anonymized_mark_0.csv" 
    dataset = load_dataset("csv", data_files=data_file_path, column_names=['decisionID', 'Text'], split="train")
    #dataset_streamed = load_dataset("csv", data_files=data_file_path, delimiter=";", column_names=['decisionID', 'Text'], split="train", streaming=True)

    #TODO: uncomment filter
    dataset = dataset.filter(lambda x: x["Text"] is not None) # remove empty rows, runtime approx 10mins

    print(dataset)
    print("####################################################################################")

    #######################    #######################    #######################

    ####################### Load models Spacy    ################################
    print("We load our NER trained models...")
    spacy.require_gpu()
    pretrained_ner_path = "model_legalbert_pretrained/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0"
    scratch_ner_path = "model_legalbert_scratch/en_pipeline-0.0.0/en_pipeline/en_pipeline-0.0.0"
    # loading the NER model, fine-tuned on my data
    NER_PRETRAINED = spacy.load(Path(pretrained_ner_path))
    NER_SCRATCH = spacy.load(Path(scratch_ner_path))
    print("...done")
    #######################    #######################    #######################

    #######################    #######################    #######################
    # some info on memory

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"Number of files in dataset : {dataset.dataset_size}")
    size_gb = dataset.dataset_size / (1024 ** 3)
    print(dataset.shape)
    print(dataset.features)
    print(f"Dataset size (cache file) : {size_gb:.2f} GB")
    #######################    #######################    #######################

    ############ to test on a sample    #######################    #################
    nrows = 1000
    #dataset = dataset.select(range(1,nrows)) 

    #######################    #######################    #######################
    print("We are now taking the data row by row using the dataset.map method...")

    # treating datasets as memory - mapped files,
    # datasets.Dataset.map() method: method to process data row by row
    
    # default batch size is 1000
    batch_size = 1000

    start = time.time()

    updated_dataset = dataset.map(inference_ner, batched=True, batch_size=batch_size)

    end = time.time()

    print(f"Inference step for dataset on: {len(dataset)} rows, {batch_size} batch size: time= ", end-start)

    # returns the last batch only
    df = updated_dataset.to_pandas()
    print(df.head(10))
    print(len(df))

    #######################    #######################    #######################
    # some info on memory

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"Number of files in dataset : {updated_dataset.dataset_size}")
    size_gb_new = updated_dataset.dataset_size / (1024 ** 3)
    print(updated_dataset.shape)
    print(updated_dataset.features)
    print(f"Dataset size (cache file) : {size_gb_new:.2f} GB")


    updated_dataset.save_to_disk("legal_prediction/law_case_text_classification/8june_dataset_entities.csv")


"""
    updated_dataset = dataset.map(lambda row:
                                  {
        'decisionID': row['decisionID'],
        "GPE": get_ent_pretrained(row, target_ent='GPE'),
        "DATE": get_ent_pretrained(row, target_ent='DATE'),
        "NORP": get_ent_pretrained(row, target_ent='NORP'),
        "ORG": get_ent_pretrained(row, target_ent='ORG'),
        "LAW": get_ent_pretrained(row, target_ent='LAW'),
        "CLAIMANT_EVENTS": get_ent_scratch(row, target_ent='CLAIMANT_EVENTS'),
        "CREDIBILITY": get_ent_scratch(row, target_ent='CREDIBILITY'),
        "DETERMINATION": get_ent_scratch(row, target_ent='DETERMINATION'),
        "CLAIMANT_INFO": get_ent_scratch(row, target_ent='CLAIMANT_INFO'),
        "PROCEDURE": get_ent_scratch(row, target_ent='PROCEDURE'),
        "DOC_EVIDENCE": get_ent_scratch(row, target_ent='DOC_EVIDENCE'),
        "EXPLANATION": get_ent_scratch(row, target_ent='EXPLANATION'),
        "LEGAL_GROUND": get_ent_scratch(row, target_ent='LEGAL_GROUND'),
        "LAW_CASE": get_ent_scratch(row, target_ent='LAW_CASE'),
        "LAW_REPORT": get_ent_scratch(row, target_ent='LAW_REPORT'),
    },
                                  remove_columns=dataset.column_names,
                                  #batched=True,
                                  #batch_size=10
                                  )
"""








