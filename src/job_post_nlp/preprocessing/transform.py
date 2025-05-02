import json
from pathlib import Path

import polars as pl
import spacy
import yaml
from tqdm import tqdm

from job_post_nlp.utils.find_project_root import find_project_root


class ColumnNotFoundError(Exception):
    def __init__(self, column_name: str) -> None:
        super().__init__(f"Column '{column_name}' not found in the Excel file.")


def load_data(file_path: Path, column_name: str = "Text") -> list:
    """
    Load data from an Excel file and extract the specified column.

    Args:
        file_path (Path): Path to the Excel file.
        column_name (str): Name of the column to extract.

    Returns:
        list: A list of texts from the specified column.
    """
    # Load the Excel file
    df = pl.read_excel(file_path, sheet_name="Sheet1")

    # Extract the specified column
    if column_name not in df.columns:
        raise ColumnNotFoundError(column_name)
    return df[column_name].to_list()


def tokenize_texts(texts: list) -> list:
    """
    Tokenize a list of texts using spaCy.

    Args:
        texts (list): A list of texts to tokenize.

    Returns:
        list: A list of tokenized texts.
    """
    # Load the spaCy language model
    nlp = spacy.load("da_core_news_sm")

    # Tokenize the texts
    tokenized_texts = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        if text is not None:  # Handle potential None values
            doc = nlp(text)
            tokens = [token.text for token in doc]
            tokenized_texts.append(tokens)

    return tokenized_texts


def export_tokens(tokens: list, output_file: Path) -> None:
    """
    Export tokenized texts to a file in JSON format.

    Args:
        tokens (list): A list of tokenized texts.
        output_file (Path): Path to the output file.
    """
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=4)


def main() -> None:
    # Define file paths
    project_root = find_project_root(__file__)
    params_path = Path(project_root) / "params.yaml"
    file_path = project_root / "data" / "Jobnet.xlsx"
    output_file = project_root / "data" / "tokenized_texts.json"

    # Load parameters
    with open(params_path) as file:
        params = yaml.safe_load(file)["preprocessing"]
    nobs = params["nobs"]

    # Process the data
    texts = load_data(file_path)
    tokenized_texts = tokenize_texts(texts[:nobs])
    export_tokens(tokenized_texts, output_file)

    print(f"Tokenized texts exported to {output_file}")


if __name__ == "__main__":
    main()
