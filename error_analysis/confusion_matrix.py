from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# file
#file = "error_analysis/df_test_predictions.csv"
file = "error_analysis/casecover_test_predictions.csv"

# load the df
df = pd.read_csv(file, sep=",")
print("--------------------------------------------")
print(df.columns)
print(df.shape)
print(df.head())    # check the data
print("--------------------------------------------")

# replace the labels by numbers in the prediciton col
df["labels"] = df["labels"].replace(0, "REJECTED")
df["labels"] = df["labels"].replace(1, "GRANTED")
df["prediction"] = df["prediction"].replace("NEGATIVE","REJECTED")
df["prediction"] = df["prediction"].replace("POSITIVE","GRANTED")
print(df.head())    # check the data


# confusion matrix
cm = confusion_matrix(df["labels"], df["prediction"])
print("--------------------------------------------")
print(cm)

# confusion matrix drawing as a heatmap
# transform in a dataframe
cm = pd.DataFrame(cm, index=["REJECTED", "GRANTED"], columns=["REJECTED", "GRANTED"])
print("--------------------------------------------")
print(cm)
# transform in a heatmap
plt.figure(figsize=(10, 10))
# change the colors to a scale of red
sns.heatmap(cm, annot=True, cmap="Reds", fmt="d")
plt.xlabel("Prediction")
plt.ylabel("Truth")
# save in pdf
plt.savefig("error_analysis/confusion_matrix_casecover_casehold.pdf")

# print the wrongly predicted samples
print("--------------------------------------------")
print("Wrongly predicted samples:")
print("--------------------------------------------")
print(df[df["labels"] != df["prediction"]])
print("--------------------------------------------")
print("Number of wrongly predicted samples:")
print("--------------------------------------------")
print(df[df["labels"] != df["prediction"]].shape[0])
# save in csv
df[df["labels"] != df["prediction"]].to_csv("error_analysis/wrongly_predicted_samples_casecover_casehold.csv", index=False)



