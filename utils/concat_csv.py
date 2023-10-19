import shutil
from glob import glob

# 20 csv files 
files = glob("anonymized_mark_*.csv")

print(f"We select {len(files)} files")


target_file_name = 'all_sentences_anonymised.csv';
shutil.copy(files[0], target_file_name)
with open(target_file_name, 'a') as out_file:
    for source_file in files[1:]:
        with open(source_file, 'r') as in_file:
#             if your csv doesn't contains header, then remove the following line.
            in_file.readline()
            shutil.copyfileobj(in_file, out_file)
            in_file.close()
    out_file.close()
    print("all csv merged, written to one .csv")
