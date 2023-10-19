# Compare?
# - the length of the text
# - the number of SEP tokens (proxy: average number of entity types)
# - the words that occur most often in the text column

import pandas as pd

f = "error_analysis/wrongly_predicted_samples_casecover_casehold.csv"

######## to compare with the rightly predicted samples
f_right = "error_analysis/casecover_test_predictions.csv"

df_wrong = pd.read_csv(f, sep=",")
df_right = pd.read_csv(f_right, sep=",")

print("df_wrong:")
print(df_wrong.shape)
print("df_right:")
print(df_right.shape)

# in the column text, average length    
df_wrong["len_text"] = df_wrong["text"].str.split().apply(lambda x: len(x))
average_length = df_wrong["len_text"].mean()
# same for the rightly predicted samples
df_right["len_text"] = df_right["text"].str.split().apply(lambda x: len(x))
average_length_right = df_right["len_text"].mean()
print("--------------------------------------------")
print(f"Average length of the text in the misclassified samples: {average_length}")
print(f"Average length of the text for all samples: {average_length_right}")
print("--------------------------------------------")

# in the column text, average compute the number of SEP tokens
df_wrong["sep_tokens"] = df_wrong["text"].apply(lambda x: x.count("[SEP]"))
average_sep = df_wrong["sep_tokens"].mean()
# same for the rightly predicted samples
df_right["sep_tokens"] = df_right["text"].apply(lambda x: x.count("[SEP]"))
average_sep_right = df_right["sep_tokens"].mean()
print("--------------------------------------------")
print(f"Average number of SEP tokens (proxy: average number of entity types) for the misclassified samples: {average_sep}")
print(f"Average number of SEP tokens (proxy: average number of entity types) for all samples: {average_sep_right}")
print("--------------------------------------------")

# words that occur most often in the text column
print("--------------------------------------------")
print("Words that occur most often in the text column, for misclassified samples:")
print("--------------------------------------------")
print(df_wrong["text"].str.split(expand=True).stack().value_counts()[:50])
print("--------------------------------------------")
print("Words that occur most often in the text column, for all samples:")
print("--------------------------------------------")
print(df_right["text"].str.split(expand=True).stack().value_counts()[:50])
print("--------------------------------------------")






