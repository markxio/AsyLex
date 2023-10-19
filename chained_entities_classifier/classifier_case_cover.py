# Text classification binary
# using PyTorch
# and hugging face

# to fix freaking cache_dir problems
# by default HF tries to default to ~/.cache/huggingface/modules
# on cirrus, the /home directory is not available for compute nodes
# therefore need to change the default cache dir before loading any modules


import os
os.environ['HF_HOME'] = f"{cache_dir_base}/huggingface/"
os.environ['HF_MODULES_CACHE'] = f"{cache_dir_base}/huggingface/modules/"
os.environ['TRANSFORMERS_CACHE'] = f"{cache_dir_base}/transformers/"
os.environ['XDG_CACHE_HOME'] = f"{cache_dir_base}/huggingface/accelerate/"
os.environ['HUGGINGFACE_ASSETS_CACHE'] = f"{cache_dir_base}/huggingface/assets/"
os.environ['TRITON_CACHE_DIR'] = f"{cache_dir_base}/triton/"
os.environ['WANDB_DISABLED'] = 'true'


import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding, AutoConfig
import evaluate
from evaluate import load
from evaluate.utils.file_utils import DownloadConfig
import re
import time
from pynvml import *

def clean_str(input_str):
    output_str = re.sub('[^\w\s\[\]]', ' ', input_str)
    output_str = re.sub(' +', ' ', output_str)
    return output_str

def get_dataset(file_path_case_cover):
    print("")
    print("#########################################################################")
    print("### Preparing the data")
    print("#########################################################################")
    print("")

    df = pd.read_csv(file_path_case_cover, sep=";")

    df.drop(['Unnamed: 0', 'index', 'text_case_cover'], inplace=True, axis=1)
    uniques = df["decisionID"].unique()
    print(len(uniques))
    df = df[df.decision_outcome != 2]  # delete "uncertain" rows

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

    col_entities = ['extracted_dates', 'LOC_HEARING','TRIBUNAL', 'PUBLIC_PRIVATE_HEARING', 'INCHAMBER_VIRTUAL_HEARING', 'JUDGE', 'DATE_DECISION']
    df = df.fillna('')
    df["text"] = df[col_entities].apply("[SEP]".join, axis=1)


    # now we can drop all useless col and only keep text, decisionID and outcome
    df.drop(['extracted_dates', 'LOC_HEARING','TRIBUNAL', 'PUBLIC_PRIVATE_HEARING', 'INCHAMBER_VIRTUAL_HEARING', 'JUDGE', 'DATE_DECISION'],
            axis=1, inplace=True)
    df.rename({"decision_outcome": "labels"}, axis=1, inplace=True)
    df["labels"] = df["labels"].astype(int)

    df["text"] = df["text"].apply(clean_str)

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
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8,16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.02),
        "num_train_epochs": trial.suggest_int("num_train_epochs",1,4)

    }


if __name__ == '__main__':

    start_time = time.time()

    TRANSFORMERS_OFFLINE=1
    HF_DATASETS_OFFLINE=1

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
    MODEL_NAME = "casehold/legalbert"
    #config = AutoConfig.from_pretrained(f"plms/{MODEL_NAME}/config.json")

    MAX_LEN = 512
    # MAX_LEN specifies the length of each tokenized sentence
    # If a tokenized text is shorter than MAX_LEN, then it is padded with zeros
    # If a tokenized text is greater than MAX_LEN, then it is truncated (this is why we have truncation=True and padding=True)

#    (vocab_size
#     = 30522max_position_embeddings = 512sinusoidal_pos_embds = Falsen_layers = 6n_heads = 12dim = 768hidden_dim = 3072dropout = 0.1attention_dropout = 0.1activation = 'gelu'initializer_range = 0.02qa_dropout = 0.1seq_classif_dropout = 0.2pad_token_id = 0 ** kwargs)

    ###################################################################
    # Dataset    
    # CASE COVER ENTITIES
    #file_path_all = "/home/clairebarale/PycharmProjects/legal_prediction/data/main_and_case_cover_12june.csv"

    # print_gpu_utilization()
    # file_path_case_cover = "case_cover_entities_decision_outcome_9june.csv"

    # Make numpy values easier to read
    np.set_printoptions(precision=3, suppress=True)
    #df = get_dataset(file_path_case_cover)

    # create test/train sets
    #df = shuffle(df)
    #size_df = len(df)
    #size_test_set = 0.2
    #split = int(size_df * size_test_set)

    #df[:split].reset_index(drop=True).to_csv('./classifier_feature_based/df_test_covers.csv', index=False)
    #df[split:].reset_index(drop=True).to_csv('./classifier_feature_based/df_train_covers.csv', index=False)
    #df_test = pd.read_csv('./classifier_feature_based/df_test_covers.csv')
    #df_train = pd.read_csv('./classifier_feature_based/df_train_covers.csv')
    #print(f"Number of example test set: {len(df_test)}")
    #print(f"Number of example train set: {df_train.shape}")


    print("script started...")
    # https://huggingface.co/docs/datasets/v1.11.0/_modules/datasets/load.html#load_dataset
    # on cirrus, we have to use the /work/... directory
    # by default, load_dataset() falls back to ~/datasets which is in /home/...
    # therefore, have to set cache_dir arg manually to our newly created huggingface cache dir
    # here: /work/sc114/sc114/clairepaola/.huggingface_cache


    dataset = load_dataset('csv', data_files={'train': 'data/df_train_covers.csv',
                                              'test': 'data/df_test_covers.csv'},)
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(f"sentences_classifier_BERT_based/{MODEL_NAME}")


    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_data = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create a map of the expected ids to their label
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(f"sentences_classifier_BERT_based/{MODEL_NAME}", num_labels=2, id2label=id2label, label2id=label2id).to("cuda")

    print_gpu_utilization()


    training_args = TrainingArguments(
        output_dir=f"checkpoints_casecover_{MODEL_NAME}",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        learning_rate=9.86e-06,
        #auto_find_batch_size=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        #gradient_accumulation_steps=16,
        #gradient_checkpointing = True,
        #fp16=True,
        weight_decay=0.015,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_on_each_node=True,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(
                f"plms/{MODEL_NAME}",
                num_labels=2,
                id2label=id2label,
                label2id=label2id
                )

    HF_DATASETS_OFFLINE=1

    accuracy = evaluate.load(path="classifier_feature_based/accuracy.py")
    f1_metric = evaluate.load(path="classifier_feature_based/f1.py", cache_dir=".huggingface_cache/evaluate")
    print("loading evaluation metrics...")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        weighted_f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "macro_f1": macro_f1["f1"], "weighted_f1":weighted_f1["f1"]}

    print("---------------------------")
    print("Evalutation metrics loaded!")
    print("---------------------------")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        #model_init = model_init
    )

    print("Training started")
    result = trainer.train()

    # to save the model locally:
    trainer.save_model(f"saved_models_casecover/{MODEL_NAME}")


    # uncomment for hyperparameter search
    #best_trial = trainer.hyperparameter_search(
    #direction="maximize",
    #backend="optuna",
    #hp_space=optuna_hp_space,
    #)
    #print(best_trial)

    print_summary(result)
    print(result)
    print("---------------------------")
    print(MODEL_NAME)
    print("--- %s seconds ---" % (time.time() - start_time))

