# Text classification binary
# using PyTorch
# and hugging face

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding, AutoConfig
import evaluate
from RF_binary_classifier_outcome import *
from evaluate import load
import re
from pynvml import *

def clean_str(input_str):
    output_str = re.sub('[^\w\s\[\]]', ' ', input_str)
    output_str = re.sub(' +', ' ', output_str)
    output_str = re.sub('(\[SEP\]){2,}', '[SEP]', output_str)
    return output_str

def get_dataset(file_path):
    print("")
    print("#########################################################################")
    print("### Preparing the data")
    print("#########################################################################")
    print("")

    df = pd.read_csv(file_path, sep=";")
    print(len(df))
    print(df)
    print(df.columns)
    
    df = df[df.decision_outcome != 2]  # delete "uncertain" rows
    
    df.drop(['Unnamed: 0', 'CLAIMANT_EVENTS', 'Text', 'text_case_cover'], inplace=True, axis=1)
    
    uniques = df["decisionID"].unique()
    print("number of cases: ")
    print(len(uniques))

    # Let's look at the input dataset imbalance:
    neg, pos = np.bincount(df['decision_outcome'])
    total = neg + pos
    print("-----------------------------------------------------")
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    print("-----------------------------------------------------")

    ############ get the df ready: we concatenate all col to form one col of text per case
    # we put a separation token btw each col: "[SEP]"

    df = df.convert_dtypes(convert_string=True)
    print(df.dtypes)
    
    col_entities = ['GPE', 'DATE', 'NORP', 'ORG', 'LAW','CREDIBILITY', 'DETERMINATION', 'CLAIMANT_INFO',
       'PROCEDURE', 'DOC_EVIDENCE', 'EXPLANATION', 'LEGAL_GROUND', 'LAW_CASE',
       'LAW_REPORT', 'extracted_dates', 'LOC_HEARING',
       'TRIBUNAL', 'PUBLIC_PRIVATE_HEARING', 'INCHAMBER_VIRTUAL_HEARING',
       'JUDGE', 'DATE_DECISION']
    df = df.fillna('') # replace nan with empty string
    df["text"] = df[col_entities].apply("[SEP]".join, axis=1)


    # now we can drop all useless col and only keep text, decisionID and outcome
    df.drop(col_entities, axis=1, inplace=True)
    df.rename({"decision_outcome": "labels"}, axis=1, inplace=True)
    df["labels"] = df["labels"].astype(int) # cast to int
    df["text"] = df["text"].apply(clean_str) # remove punctuation 

    print(df.head(10))
    return df
    print("Done!")


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def optuna_hp_space(trial): return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8,16, 32, 64, 128]),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.02),
        "num_train_epochs": trial.suggest_int("num_train_epochs",1,4)

    }
    


if __name__ == '__main__':

    TRANSFORMERS_OFFLINE=1


    MODELS = [
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-base",
        "nlpaueb/legal-bert-base-uncased",
        "pile-of-law/legalbert-large-1.7M-2",
        "casehold/legalbert"
              ]

    ###################################################################
    # Hugging face implementation
    MODEL_NAME = 'bert-base-uncased'
    config = AutoConfig.from_pretrained(f"classifier_feature_based/{MODEL_NAME}/config.json")

    MAX_LEN = 512

    ###################################################################
    # Dataset    
    # ENTITIES
    file_path = "main_and_case_cover_12june.csv"

    #print_gpu_utilization()

    # Make numpy values easier to read
    np.set_printoptions(precision=3, suppress=True)

    #f = get_dataset(file_path)
    
    # create test/train sets
    #df = shuffle(df)
    #size_df = len(df)
    #size_test_set = 0.2
    #split = int(size_df * size_test_set)

    #df[:split].reset_index(drop=True).to_csv('./classifier_feature_based/df_test_all.csv', index=False)
    #df[split:].reset_index(drop=True).to_csv('./classifier_feature_based/df_train_all.csv', index=False)
    #df_test = pd.read_csv('./classifier_feature_based/df_test_all.csv')
    #df_train = pd.read_csv('./classifier_feature_based/df_train_all.csv')
    #print(f"Number of example test set: {len(df_test)}")
    #print(f"Number of example train set: {df_train.shape}")

    dataset = load_dataset('csv', data_files={'train': './classifier_feature_based/df_train_all.csv',
                                              'test': './classifier_feature_based/df_test_all.csv'})
    print(dataset)


    # TOKENIZATION
    tokenizer = AutoTokenizer.from_pretrained(f"classifier_feature_based/{MODEL_NAME}")


    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_data = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create a map of the expected ids to their label
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(f"classifier_feature_based/{MODEL_NAME}", num_labels=2, id2label=id2label, label2id=label2id).to("cuda")

    #print_gpu_utilization()

    training_args = TrainingArguments(
        output_dir=f"./classifier_feature_based/model_allents_{MODEL_NAME}",
        learning_rate=6e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        #gradient_accumulation_steps=16,
        #gradient_checkpointing = True,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
        

    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(
                f"classifier_feature_based/{MODEL_NAME}",
                num_labels=2,
                id2label=id2label,
                label2id=label2id
                )

    
    HF_DATASETS_OFFLINE=1

    accuracy = evaluate.load("classifier_feature_based/accuracy.py")
    f1_metric = evaluate.load("classifier_feature_based/f1.py")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        weighted_f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "macro_f1": macro_f1["f1"], "weighted_f1":weighted_f1["f1"]}


    trainer = Trainer(
        #model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        model = model_init
    )

    #results = trainer.train()

    best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    )

    print(best_trial)
    print(MODEL_NAME)
    #print_summary(result)
