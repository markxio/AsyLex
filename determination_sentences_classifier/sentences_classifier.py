# Determination sentences Classifier (Table titled: Determination extracted sentences)
# Text classification with distilBERT
# using Tensorflow: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
# and hugging face library : https://github.com/peterbayerle/huggingface_notebook/blob/main/distilbert_tf.ipynb
# This script trains a model_HF_distilbert_sentences_clf to get the outcome of decisions
# The training data is the spans of text extracted with the "determination" flag in the NER


cache_dir_base = "/work/sc114/sc114/clairepaola/.cache"

import os
os.environ['HF_HOME'] = f"{cache_dir_base}/huggingface/"
os.environ['HF_MODULES_CACHE'] = f"{cache_dir_base}/huggingface/modules/"
os.environ['TRANSFORMERS_CACHE'] = f"{cache_dir_base}/transformers/"
os.environ['XDG_CACHE_HOME'] = f"{cache_dir_base}/huggingface/accelerate/"
os.environ['HUGGINGFACE_ASSETS_CACHE'] = f"{cache_dir_base}/huggingface/assets/"
os.environ['TRITON_CACHE_DIR'] = f"{cache_dir_base}/triton/"
os.environ['WANDB_DISABLED'] = 'true'


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle, resample
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate



# Clean the dataframe, delete empty rows and first col
def clean_raw_data(df):
    df.dropna(axis=0, inplace=True)
    return df

# We preprocess the df to get it ready for training
# convert sentences from list to str
# convert decision from float64 to integer
def clean_df(df):
    df["decision"] = df["decision"].astype('int')
    df['extracted_sentences_determination'] = df['extracted_sentences_determination'].astype('string')
    df['extracted_sentences_determination'] = [s.replace('[\'', '') for s in df['extracted_sentences_determination']]
    df['extracted_sentences_determination'] = [s.replace('\']', '') for s in df['extracted_sentences_determination']]
    return df

def plot_class_distibution(df):
    colors = plt.cm.Accent(np.linspace(0, 1, num_classes))
    iter_color = iter(colors)
    df['decision'].value_counts().plot.barh(title="Sentences per label (n, %)",
                                            ylabel="Decision outcome",
                                            color=colors,
                                            figsize=(9, 9))

    for i, v in enumerate(df['decision'].value_counts()):
        c = next(iter_color)
        plt.text(v, i,
                 " " + str(v) + ", " + str(round(v * 100 / df.shape[0], 2)) + "%",
                 color=c,
                 va='center',
                 fontweight='bold')
    history_file_name = "plot_class_training_distribution.pdf"
    plt.savefig(history_file_name, bbox_inches='tight')

def get_num_words_per_sample(sample_sentence):
    num_words = len(sample_sentence.split())
    return num_words

def plot_history_loss_accuracy(loss, val_loss, acc, val_acc):
    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss') # r is for "solid red line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss') # b is for "solid blue line"
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    history_file_name = "plot_history_loss_accuracy.pdf"
    plt.savefig(history_file_name)
    print(f"Training history with accuracy written to file %s" % history_file_name)

def get_df_for_inference(df):
    df_all_sentences = df.copy() # create a copy to keep all sentences for inference step later on
    df_all_sentences['extracted_sentences_determination'] = df_all_sentences['extracted_sentences_determination'].astype('string')
    df_all_sentences['extracted_sentences_determination'] = [s.replace('[\'', '') for s in df_all_sentences['extracted_sentences_determination']]
    df_all_sentences['extracted_sentences_determination'] = [s.replace('\']', '') for s in df_all_sentences['extracted_sentences_determination']]
    df_all_sentences.drop("decision", axis=1, inplace=True)
    return df_all_sentences


if __name__ == '__main__':

    TRANSFORMERS_OFFLINE=1

    # Make numpy values easier to read
    np.set_printoptions(precision=3, suppress=True)

    ################### get the data from CSV file
    print("")
    print("#########################################################################")
    print("### Preparing the data")
    print("#########################################################################")
    print("")


    ############ Uncomment to get the data from scratch here:  
    #csv_file_path = "./data/determination_data/sentences+decision_label.csv"
    # first column is the text of the sentences
    # second column is the label

    #df = pd.read_csv(csv_file_path, sep=';', index_col=None, encoding='latin-1')
    #raw_df = df.shape
    #print(df.shape)
    #df_all_sentences = get_df_for_inference(df)

    #df = clean_raw_data(df)
    #new_df = df.shape
    #number_rows = len(df)
    #df.to_csv("gold_annotated_determination_sentences.csv", sep=";")
    #print(len(df))
    #uniques = df["decisionID"].unique()
    #print("-------------------------------------------------------------------")
    #print("We load the data from CSV file to df. Df contains {} before cleaning, and {} after removing unlabelled rows".format(raw_df, new_df))
    #print(f"We have {len(uniques)} rows ie cases")
    #print("-------------------------------------------------------------------")
    #df = clean_df(df)

    #df.drop("decisionID", axis=1, inplace=True)
    #print("-----------------------------------------------------")
    #print(df.head())
    #print("-----------------------------------------------------")

    # it is a binary classifier but just creating the variable + plot of class distribution
    #num_classes = len(df["decision"].value_counts())
    #plot_class_distibution(df)

    # Let's look at the input dataset imbalance:
    #neg, pos = np.bincount(df['decision'])
    #total = neg + pos
    #print("-----------------------------------------------------")
    #print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    #    total, pos, 100 * pos / total))
    #print("-----------------------------------------------------")

    # returns the median number of words per sample
    #myarray = []
    #for sentence in df['extracted_sentences_determination']:
    #    num_words = get_num_words_per_sample(sentence)
    #    myarray.append(num_words)
    #median_num_words = np.median(myarray)
    #mean_num_words = np.mean(myarray)
    #print("The median number of words in the input samples is: {} \n"
    #      "The mean number of words in the input samples is {}".format(median_num_words, mean_num_words))

    #df.rename(columns={"extracted_sentences_determination": "text", "decision": "labels"}, inplace=True)

    # Hugging face implementation
    MODELS = [
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-base",
        "nlpaueb/legal-bert-base-uncased",
        "pile-of-law/legalbert-large-1.7M-2",
        "casehold/legalbert",
        "lexlms/legal-roberta-large"
    ]

    MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 100
        # MAX_LEN specifies the length of each tokenized sentence
        # If a tokenized review is shorter than MAX_LEN, then it is padded with zeros
        # If a tokenized review is greater than MAX_LEN, then it is truncated (this is why we have truncation=True and padding=True

    # Dataset
    #df = shuffle(df)
    #size_df = len(df)
    #size_test_set = 0.2
    #split = int(size_df*size_test_set)

    #df[:split].reset_index(drop=True).to_csv('df_test.csv', index=False)
    #df[split:].reset_index(drop=True).to_csv('df_train.csv', index=False)
    df_test = pd.read_csv('df_test.csv')
    df_train = pd.read_csv('df_train.csv')

    # Oversampling: on the df_train
    #df_majority = df_train[(df_train["labels"] == 0)]
    #df_minority = df_train[(df_train["labels"] == 1)]
    #df_copy = df_train[(df_train["labels"] == 1)]
    #df_copy2 = df_train[(df_train["labels"] == 1)]
    #df_copy3 = df_train[(df_train["labels"] == 1)]
    #df_copy4 = df_train[(df_train["labels"] == 1)]
    #df_upsampling = pd.concat([df_minority, df_copy, df_copy2, df_copy3, df_copy4])
    #df_train = pd.concat([df_upsampling, df_majority])
    #df_train.to_csv('df_train.csv', index=False)
    #df_train = pd.read_csv('df_train.csv')

    print(df_train)
    print(df_test)
    print("--------------------------------------")
    print(df_train['labels'].value_counts())
    print(df_test['labels'].value_counts())


    neg, pos = np.bincount(df_train['labels'])
    total = neg + pos
    print("-----------------------------------------------------")
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    print("-----------------------------------------------------")


    dataset = load_dataset('csv', data_files={'train': 'df_train.csv',
                                              'test': 'df_test.csv'})
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(f"sentences_classifier_BERT_based/{MODEL_NAME}")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_data = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create a map of the expected ids to their label
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(f"sentences_classifier_BERT_based/{MODEL_NAME}", num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"./sentences_classifier_{MODEL_NAME}/9oct_model_2epochs",
        overwrite_output_dir=True,
        learning_rate=2e-5,
        auto_find_batch_size=True,
        #per_device_train_batch_size=16,
        #per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    accuracy = evaluate.load(path="classifier_feature_based/accuracy.py")
    f1_metric = evaluate.load(path="classifier_feature_based/f1.py")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        weighted_f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "macro_f1": macro_f1["f1"], "weighted_f1":weighted_f1["f1"]}


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    save_model_path = f"./save_models_{MODEL_NAME}/"
    trainer.save_model(save_model_path)







