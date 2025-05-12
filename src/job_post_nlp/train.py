import json
import os
import pathlib
from collections import Counter
from pathlib import Path

import spacy
import yaml
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

from job_post_nlp.utils.find_project_root import find_project_root


def load_corpus(file_path: Path) -> DocBin:
    """
    Load a preprocessed corpus from a .spacy binary file.
    Args:
        file_path (Path): Path to the .spacy file.
    Returns:
        DocBin: The loaded corpus.
    """
    doc_bin = DocBin().from_disk(file_path)
    return doc_bin


def unpack_corpus(corpus: DocBin) -> list[tuple[str, list[str]]]:
    """
    Unpack the corpus and return a list of tuples containing text IDs and their corresponding lemmas.
    Args:
        corpus (DocBin): The preprocessed corpus.
    Returns:
        list: A list of tuples containing text IDs and their corresponding lemmas.
    """
    nlp = spacy.blank("da")
    if not Doc.has_extension("text_id"):
        Doc.set_extension("text_id", default=None)

    return [
        (doc._.text_id, list(get_clean_tokens(doc)))
        for doc in tqdm(
            corpus.get_docs(nlp.vocab),
            total=corpus.__len__(),
            desc="Unpacking texts",
        )
    ]


def get_clean_tokens(doc: Doc) -> list[str]:
    """
    Clean the tokens by removing non-alphabetic characters and stop words.
    Args:
        doc (Doc): A spaCy Doc object.
    Returns:
        list: A list of cleaned tokens.
    """
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]


def get_most_common_words(corpus: DocBin, params: dict) -> list:
    """
    Get the most common words from a corpus of tokenized texts.

    Args:
        corpus (list[list[str]]): A list of tokenized texts.
        params (dict): A dictionary containing parameters, including 'top_n'.

    Returns:
        list: A list of tuples containing the most common words and their counts.
    """
    # Unpack parameters
    top_n = params["top_n"]

    # Flatten the list of tokenized texts
    all_tokens = [token for _, tokens in unpack_corpus(corpus) for token in tokens]

    # Count the occurrences of each word
    word_counts = Counter(all_tokens)

    # Get the most common words
    return word_counts.most_common(top_n)


def export_most_common_words(common_words: list, output_file: str | pathlib.Path) -> None:
    """
    Export the most common words to a file in JSON format.

    Args:
        common_words (list): A list of tuples containing words and their counts.
        output_file (str): Path to the output file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(common_words, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    params_path = project_root / "params.yaml"
    corpus_file = data_dir / "corpus.spacy"  # Use corpus file
    output_file = data_dir / "most_common_words.json"

    # Load parameters
    with open(params_path) as file:
        params = yaml.safe_load(file)["train"]

    # Process
    corpus = load_corpus(corpus_file)  # Load corpus instead of tokens
    common_words = get_most_common_words(corpus, params)
    export_most_common_words(common_words, output_file)

    print(f"Most common words exported to {output_file}")
