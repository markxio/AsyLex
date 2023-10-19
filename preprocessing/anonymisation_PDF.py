"""
Script to anonymise the data
Output: text with names replaces by XX
Tool used: Predisio analyzer and anonymizer (Microsoft)
https://github.com/microsoft/presidio
"""
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider, SpacyNlpEngine
from pathlib import Path
import glob
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import multiprocessing
import numpy as np

def anonymize_text(text):
    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # Pass the created NLP engine and supported_languages to the AnalyzerEngine
    main_analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    # get the results from the analyzer (i.e. the flagged entities)
    analyzer_results = main_analyzer.analyze(text=text, entities=["PERSON"], language="en")
    #print(analyzer_results)
    analyzer_results = analyzer_results

    # Initialize the engine:
    anonymizer = AnonymizerEngine()

    # Invoke the anonymize function with the text,
    # analyzer results (potentially coming from presidio-analyzer) and
    # Operators to get the anonymization output:
    # Define anonymization operators
    operators = {
            "PERSON": OperatorConfig(
                "mask",
                {
                    "type": "mask",
                    "masking_char": "X",
                    "chars_to_mask": 20,
                    "from_end": True,
                }
            )
        }

    anonymized_results = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators
    )

    #print(anonymized_results)
    #print(f"text: {anonymized_results.text}")
    return anonymized_results.text

def process_chunk(txt_files, chunk_id):
    for file in txt_files: #tqdm(txt_files):
        with open(file, encoding='utf8') as f:
            #get the file name
            file_name = os.path.basename(file)
            file_name = Path(file_name).stem  # removing the .txt
            output_file = open(f"case_text_anonymized/{file_name}.txt",'w')

            #print(f"process_chunk: chunk_id {chunk_id} file_name is {file_name}")

            # open the file and get the text
            text = f.read()
            
            #print(f"process_chunk: chunk_id {chunk_id} file_name is {file_name} -- text has been read!")

            # replace names with x values
            anonymized_text = anonymize_text(text)
            
            #print(f"process_chunk: chunk_id {chunk_id} file_name is {file_name} -- text anonymized!")

            # write to new text file
            print(anonymized_text, file=output_file)

            print(f"process_chunk: chunk_id {chunk_id} file_name is {file_name} -- file written!")

            output_file.close()


if __name__ == '__main__':
    # Text to anonymize
    #text = "The judge Miller says the claimant is 26 years old nigerian, James Bond"
    source_folder = "case_text/cases_text"

    source_folder = Path(source_folder).expanduser()
    txt_files = glob.glob(f"{source_folder}/*.txt")
    print(f"-- Folder {source_folder} holds {len(txt_files)} .txt files")

    #n_cores = multiprocessing.cpu_count()
    n_cores = 1
    print(f"running in parallel on {n_cores} cores...")
    files_list_chunks = np.array_split(txt_files, n_cores)
    print(files_list_chunks)
    Parallel(n_jobs=n_cores)(delayed(process_chunk)(files_list_chunks[i], i) for i in range(0, n_cores))













