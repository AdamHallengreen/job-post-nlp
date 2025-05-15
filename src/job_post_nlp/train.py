import json
import os
import pathlib
from pathlib import Path

import polars as pl
import yaml
from spacy.tokens import DocBin

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


def load_tdm(file_path: Path) -> pl.DataFrame:
    """
    Load a Term Document Matrix (TDM) from a CSV file.
    Args:
        file_path (Path): Path to the TDM CSV file.
    Returns:
        pl.DataFrame: The loaded TDM.
    """
    if not file_path.exists():
        raise FileNotFoundError()
    return pl.read_parquet(file_path)


def get_most_common_words(tdm: pl.DataFrame, params: dict) -> dict:
    """
    Get the most common words from a term-document matrix (TDM).

    Args:
        tdm (pl.DataFrame): The term-document matrix with columns as terms and rows as documents.
        params (dict): A dictionary containing parameters, including 'top_n'.

    Returns:
        dict: A dict mapping words to their counts for the most common words.
    """
    top_n = params["top_n"]

    # Exclude the 'doc_id' column if present
    term_columns = [col for col in tdm.columns if col != "doc_id"]
    # Sum each term column to get total frequency across all documents
    sums = tdm.select(term_columns).sum()
    # Use unpivot instead of melt (melt is deprecated)
    df_long = sums.unpivot(
        on=term_columns,
        variable_name="word",
        value_name="count",
    )
    # Sort the DataFrame by count in descending order and select the top_n rows
    df_sorted = df_long.sort("count", descending=True).head(top_n)
    # Convert to dict for JSON export
    return dict(zip(df_sorted["word"].to_list(), [int(x) for x in df_sorted["count"].to_list()]))


def export_most_common_words(common_words: dict, output_file: str | pathlib.Path) -> None:
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
    tdm_file = data_dir / "tdm.parquet"

    # Load parameters
    with open(params_path) as file:
        params = yaml.safe_load(file)["train"]

    # Process
    tdm = load_tdm(tdm_file)
    common_words = get_most_common_words(tdm, params)
    export_most_common_words(common_words, output_file)

    print(f"Most common words exported to {output_file}")
