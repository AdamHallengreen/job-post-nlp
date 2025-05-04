import json
import os
import pathlib
from collections import Counter
from pathlib import Path

import yaml

from job_post_nlp.utils.find_project_root import find_project_root


class InvalidTokenFormatError(Exception):
    def __init__(self) -> None:
        super().__init__("Input file must contain a list of lists of strings.")


def load_tokens(input_file: str | pathlib.Path) -> list[list[str]]:
    """
    Load tokenized texts from a JSON file.

    Args:
        input_file (str | pathlib.Path): Path to the JSON file containing tokenized texts.

    Returns:
        list[list[str]]: A list of tokenized texts, where each text is a list of strings (tokens).
    """
    with open(input_file, encoding="utf-8") as f:
        tokens = json.load(f)
    if not isinstance(tokens, list) or not all(
        isinstance(text, list) and all(isinstance(token, str) for token in text) for text in tokens
    ):
        raise InvalidTokenFormatError()
    return tokens


def get_most_common_words(tokens: list, params: dict) -> list:
    """
    Get the most common words from a list of tokenized texts.

    Args:
        tokens (list): A list of tokenized texts.
        params (dict): A dictionary containing parameters, including 'top_n'.

    Returns:
        list: A list of tuples containing the most common words and their counts.
    """
    # Unpack parameters
    top_n = params["top_n"]

    # Flatten the list of tokenized texts
    all_tokens = [token for text in tokens for token in text]

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
    input_file = Path(project_root) / "data" / "tokenized_texts.json"
    output_file = Path(project_root) / "data" / "most_common_words.json"

    # Load parameters
    with open(params_path) as file:
        params = yaml.safe_load(file)["train"]

    # Process
    tokens = load_tokens(input_file)
    common_words = get_most_common_words(tokens, params)
    export_most_common_words(common_words, output_file)

    print(f"Most common words exported to {output_file}")
