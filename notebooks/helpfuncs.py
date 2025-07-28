import random
from pathlib import Path

import polars as pl
import scipy.sparse as ss
import yaml

from job_post_nlp.evaluate import load_model
from job_post_nlp.prepare import load_texts
from job_post_nlp.train import load_tdm


def load_everything() -> dict:
    project_root = Path("..")
    project_root.resolve()

    models_dir = project_root / "models"
    data_dir = project_root / "data"

    tdm_file = data_dir / "tdm.npz"
    tdm_info_file = data_dir / "tdm_info.json"

    # Process
    model = load_model(models_dir / "corex_model.pkl")
    tdm, tdm_info = load_tdm(tdm_file, tdm_info_file)
    texts = load_texts(data_dir / "texts.parquet")  # Load preprocessed texts

    data = {"model": model, "tdm": tdm, "tdm_info": tdm_info, "texts": texts}
    return data


def print_words_in_doc(ident: str, data: dict) -> None:
    """
    Get the words in a document given its ident.
    Args:
        ident (str): The ID of the document.
        tdm (ss.csr_sparse_matrix): The term-document matrix.
        tdm_info (dict): Information about the TDM, including vocabulary and IDs.
    Returns:
        list: A list of words in the document.
    """
    tdm = data["tdm"]
    tdm_info = data["tdm_info"]

    if ident not in tdm_info["ids"]:
        print(f"ID {ident} not found in TDM.")
        return
    idx = tdm_info["ids"].index(ident)
    if idx >= tdm.shape[0]:
        print(f"Index {idx} is out of bounds for TDM with shape {tdm.shape}.")
        return
    doc_vector = tdm[idx, :]
    if not ss.issparse(doc_vector):
        print(f"Document vector for ID {ident} is not sparse.")
        return
    if doc_vector.nnz == 0:
        print(f"Document vector for ID {ident} is empty.")
        return

    words = []
    for word_idx in doc_vector.indices:
        if word_idx < len(tdm_info["vocab"]):
            words.append(tdm_info["vocab"][word_idx])
        else:
            print(f"Word index {word_idx} is out of bounds for vocabulary.")
    if not words:
        print(f"No words found for ID {ident}.")

    print_list_of_words(words)


def print_text(ident: str, data: dict) -> None:
    """
    Get texts from a DataFrame based on provided IDs.

    Args:
        texts (pl.DataFrame): The DataFrame containing texts.
        ids (str|list[str]): The ID or list of IDs to filter the DataFrame.

    Returns:
        pl.DataFrame: Filtered DataFrame containing only the specified texts.
    """
    text = data["texts"].filter(pl.col("id") == ident)["text"].item()

    print_text_str(text)


def print_words_and_text(ident: str, data: dict) -> None:
    """
    Get words and texts for given IDs from the term-document matrix and texts DataFrame.

    Args:
        tdm (ss.csr_matrix): The term-document matrix.
        tdm_info (dict): Information about the TDM, including vocabulary and IDs.
        texts (pl.DataFrame): The DataFrame containing texts.
        ids (str|list[str]): The ID or list of IDs to filter.

    Returns:
        tuple: A tuple containing a list of words and a DataFrame of texts.
    """

    print(f"Text for document {ident}:")
    print_text(ident, data)

    print("Words in document:")
    print_words_in_doc(ident, data)


def print_text_str(text, length=70):
    """
    Print log text with linebreaks after every length (default 70) character but only between words
    """
    if text is None:
        print("None")
        return

    text_list = text.split()

    cum_length = 0
    insert_at = length
    for i, text_part in enumerate(text_list):
        cum_length += len(text_part)
        if cum_length > insert_at:
            text_list[i] += "\n"
            insert_at += length
    print(" ".join(text_list))


def print_list_of_words(words: list, width=8) -> None:
    """
    print a list of words, width word per line.

    """
    if not words:
        print("No words to display.")
        return
    for i in range(0, len(words), width):
        print(", ".join(words[i : i + width]))


def print_random(data: dict, n: int = 5) -> None:
    """
    Print words and texts from n random vacancies.
    Args:
        data (dict): Dictionary containing 'tdm', 'tdm_info', and 'texts'.
        n (int): Number of random vacancies to print.
    """

    tdm_info = data["tdm_info"]

    random_ids = random.sample(tdm_info["ids"], n)

    for ident in random_ids:
        print(f"\nVacancy ID: {ident}")
        print_words_and_text(ident, data)


def read_anchors_from_yaml(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)["anchors"]


def check_anchors(anchors, data):
    """
    Loops trough all anchors and checks if they are in the vocabulary of the tdm_info.
    Also print how often they occur in the tdm.
    """
    tdm_info = data["tdm_info"]
    tdm = data["tdm"]
    vocab = tdm_info["vocab"]
    counts = tdm.sum(axis=0).A1  # Get sum of each column (word) in the TDM
    for anchor in anchors:
        if isinstance(anchor, str):
            anchor = [anchor]
        if not isinstance(anchor, list):
            print(f"Anchor {anchor} is not a list or string.")
            continue
        print(f"\nChecking anchor: {anchor[0]}")
        for word in anchor:
            if word not in vocab:
                print(f"!!! '{word}' not found in vocabulary.")
                continue
            idx = vocab.index(word)
            count = counts[idx] if idx < len(counts) else 0
            print(f"'{word}' occurs {count} times in the TDM.")


def find_similar_words(word, data):
    """
    Find words similar to a given word based on the TDM.
    For now it just checks if the word is contained in other words in the vocabulary.
    Args:
        word (str): The word to find similar words for.
        data (dict): Dictionary containing 'tdm' and 'tdm_info'.
    Returns:
        list: A list of similar words.
    """
    tdm = data["tdm"]
    tdm_info = data["tdm_info"]
    counts = tdm.sum(axis=0).A1  # Get sum of each column (word) in the TDM

    if word not in tdm_info["vocab"]:
        print(f"Word '{word}' not found in vocabulary.")

    # Print similar words sorted by their counts

    similar_words = []
    for idx, vocab_word in enumerate(tdm_info["vocab"]):
        if word in vocab_word:
            similar_words.append((vocab_word, counts[idx].item()))
    similar_words.sort(key=lambda x: x[1], reverse=True)
    print(f"Similar words to '{word}':")
    for similar_word, count in similar_words:
        print(f"'{similar_word}' occurs {count} times in the TDM.")
