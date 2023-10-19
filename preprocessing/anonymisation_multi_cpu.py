from presidio_anonymizer import AnonymizerEngine, BatchAnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
from presidio_analyzer.nlp_engine import NlpEngineProvider, SpacyNlpEngine
from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine
from tqdm import tqdm
import time
import spacy
from pathlib import Path
import typer
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


def anonymize_text(text):
    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    gpu_usage = spacy.require_gpu()
    print(f"1...GPU usage: {gpu_usage}")
    #spacy.prefer_gpu()


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

def anonymise_batch(df_dict: dict, core_id: int, n_cores: int): # -> pd.DataFrame:
    print("---------------------------------------")
    print(f"Processing {core_id}/{n_cores}...")


    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    gpu_usage = spacy.require_gpu()
    print(f"1...GPU usage: {gpu_usage}")
    #spacy.prefer_gpu()


    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # Pass the created NLP engine and supported_languages to the AnalyzerEngine
    main_analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=main_analyzer)
    batch_anonymizer = BatchAnonymizerEngine()
    
    # get the results from the analyzer (i.e. the flagged entities)
    #analyzer_results = main_analyzer.analyze(text=text, entities=["PERSON"], language="en")
    analyzer_results = batch_analyzer.analyze_dict(df_dict, language="en", entities=["PERSON"])
    #print(analyzer_results)
    analyzer_results = list(analyzer_results)

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

    anonymizer_results = batch_anonymizer.anonymize_dict(
        analyzer_results=analyzer_results,
        operators=operators
    )

    #print(anonymized_results)
    #print(f"text: {anonymized_results.text}")

    results_df = pd.DataFrame(anonymizer_results)
    results_df.to_csv(f"anonymized_mark_{core_id}.csv")
    print(f"anonymized results written to csv for {core_id}/{n_cores}")
    #return pd.DataFrame(anonymizer_results)

def anonymise(df):
    print("---------------------------------------")
    print(f"Processing on GPU...")
    # selects the target col
    mydf = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'decisionID'], axis=1)
    print("mydf:")
    print(mydf.head())
    print("CSV read!")
    # how many words per sentence:
    #df["sentence_length"] = df[].str.split().apply(len)
    gpu_usage = spacy.require_gpu()
    print(f"2...GPU usage: {gpu_usage}")
    #spacy.prefer_gpu()

    df_text_anonymised = mydf['Text'].apply(anonymize_text)


    # Index(['Unnamed: 0', 'Unnamed: 0.1', 'decisionID', 'Text'], dtype='object')
    non_ano_df = df
    non_ano_df = non_ano_df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Text'], axis=1, inplace=True)
    #print(non_ano_df.columns)

    result_df = pd.concat([non_ano_df, df_text_anonymised], axis=1)
    result_df.to_csv(f"GPU_all_sentences_anonymised.csv")

if __name__ == "__main__":

    print("it's started")
    path_to_tokenised_sentences = "final_sentences_ID.csv"
    
    n_cores = multiprocessing.cpu_count()
    print(f"running in parallel on {n_cores} cores...")


    df = pd.read_csv(path_to_tokenised_sentences, sep=';')
    print(df.head())
    print(len(df))
    df=df.dropna()
    print("we deleted empty rows")
    print(len(df))
    print(df.columns)

    print("calling the function now")
    
    #start = time.time()
    #anonymise(df)
    #end = time.time()
    #
    #print(f"single anonymizer for {nrows} rows: time=", end-start)
    
    chunks_df = np.array_split(df, n_cores)
    #print(chunks_df)
    
    start = time.time()
    
    #partial_results = Parallel(n_jobs=n_cores)(delayed(anonymise_batch)(chunks_df[i].to_dict(orient="list"), i, n_cores) for i in range(0, n_cores))
    for i in range(0, n_cores):
        anonymise_batch(chunks_df[i].to_dict(orient="list"), i, n_cores)

    #anonymise_batch(df.to_dict(orient="list"))
    end = time.time()
    
    print(f"batch anonymizer for {len(df)} rows: time=", end-start)

