from transformers import AutoModel, pipeline, AutoTokenizer, BertForSequenceClassification
import pandas as pd

# files
test_file = "error_analysis/df_test.csv"
train_file = "df_train.csv"

# concatenate the 2 files
df = pd.read_csv(test_file, sep=",")
df = df.sample(frac=1).reset_index(drop=True)

#df = pd.read_csv(train_file, sep=",")
#df = pd.concat([pd.read_csv(train_file), pd.read_csv(test_file)])

print("--------------------------------------------")
print(df.columns)
print(df.shape)
print(df.head())    # check the data
print("--------------------------------------------")

# load model
model_path = "models_det_sentences/save_models_casehold/legalbert/" # the best performing model with legal pretraining 

tokenizer = AutoTokenizer.from_pretrained(model_path, padding='max_length', truncation=True)
model = BertForSequenceClassification.from_pretrained(model_path)
print("Model loaded")

# predict the class of each sentence in the df now, using the model
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# take the 5 first samples in df to test
#df = df.head(10)

df["prediction"] = df["text"].apply(lambda x: classifier(x)[0]["label"])
print("--------------------------------------------")
print(df.head(10))
print("--------------------------------------------")

# save the df
df.to_csv("error_analysis/df_test_predictions.csv", index=False)
print("df saved")




