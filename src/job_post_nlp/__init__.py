import spacy

# This is just a hack to make uv understand that the dependency is not redundant
nlp = spacy.load("da_core_news_lg")
nlp = spacy.load("da_core_news_sm")
