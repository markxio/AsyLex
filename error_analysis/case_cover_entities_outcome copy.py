from transformers import AutoModel, pipeline, AutoTokenizer, BertForSequenceClassification
import pandas as pd

# files
test_file = "data/case_cover/df_test_covers.csv"
train_file = "data/case_cover/df_train_covers.csv"


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
model_path = "casehold/legalbert" # the best performing model with legal pretraining 

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
df.to_csv("error_analysis/casecover_test_predictions.csv", index=False)
print("df saved")




