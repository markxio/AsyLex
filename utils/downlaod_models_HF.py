from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


MODELS = [
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-base",
        "nlpaueb/legal-bert-base-uncased",
        "pile-of-law/legalbert-large-1.7M-2",
        "casehold/legalbert", 
        "lexlms/legal-roberta-large"
              ]


MODEL_NAME = "casehold/legalbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(f"./plms/{MODEL_NAME}")
model.save_pretrained(f"./plms/{MODEL_NAME}")

