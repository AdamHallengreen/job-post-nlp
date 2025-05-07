import json
import os
import pathlib
from collections import Counter
from pathlib import Path
from typing import cast

import yaml

from job_post_nlp.utils.find_project_root import find_project_root


class InvalidCorpusError(Exception):
    def __init__(self) -> None:
        super().__init__("The corpus must be a list of lists of strings.")


def load_corpus(input_file: str | pathlib.Path) -> list[list[str]]:
    """
    Load the preprocessed corpus from a JSON file.

    Args:
        input_file (str | pathlib.Path): Path to the JSON file containing the corpus.

    Returns:
        list[list[str]]: A list of tokenized texts, where each text is a list of strings (tokens).
    """
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    # Ensure the data is a list of lists of strings
    if not isinstance(data, list) or not all(
        isinstance(text, list) and all(isinstance(token, str) for token in text) for text in data
    ):
        raise InvalidCorpusError()
    corpus: list[list[str]] = cast(list[list[str]], data)  # Explicitly cast to the expected type
    return corpus


def get_most_common_words(corpus: list[list[str]], params: dict) -> list:
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
    all_tokens = [token for text in corpus for token in text]

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
    project_root = find_project_root(__file__)
    params_path = Path(project_root) / "params.yaml"
    corpus_file = Path(project_root) / "data" / "corpus.json"  # Use corpus file
    output_file = Path(project_root) / "data" / "most_common_words.json"

    # Load parameters
    with open(params_path) as file:
        params = yaml.safe_load(file)["train"]

    # Process
    corpus = load_corpus(corpus_file)  # Load corpus instead of tokens
    common_words = get_most_common_words(corpus, params)
    export_most_common_words(common_words, output_file)

    print(f"Most common words exported to {output_file}")
