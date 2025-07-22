import json
import math
import os
from collections.abc import Generator
from pathlib import Path
from typing import Optional

import polars as pl
import scipy.sparse as ss
import spacy
from lingua import LanguageDetectorBuilder
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

from job_post_nlp.utils.interactive import try_inter

try_inter()
from job_post_nlp.utils.find_project_root import find_project_root  # noqa: E402


class FileNotFoundErrorMessage:
    def __init__(self, file_path: Path) -> None:
        self.message = f"File {file_path} does not exist."

    def __str__(self) -> str:
        return self.message


class UnsupportedFileTypeError(Exception):
    def __init__(self, file_suffix: str) -> None:
        self.message = f"Unsupported file type: {file_suffix}"
        super().__init__(self.message)


class UnsupportedLinguaOutput(Exception):
    def __init__(self, output: str) -> None:
        self.message = f"Unsupported Lingua output: {output}"
        super().__init__(self.message)


def load_data(file_path: Path, par: DictConfig) -> pl.DataFrame:
    """
    Load data from an Excel file.

    Args:
        file_path (Path): Path to the Excel file.
        par (DictConfig): Configuration parameters.

    Returns:
        pl.DataFrame: DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedFileTypeError: If the file is not an Excel file.
    """
    # Check if using STAR data
    if par.star.usestar:
        return load_star_data()

    # check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(FileNotFoundErrorMessage(file_path))
    # check if the file is an Excel file
    if file_path.suffix in [".xlsx", ".xls"]:
        df = load_excel(file_path)
    else:
        raise UnsupportedFileTypeError(file_path.suffix)
    return df


def load_star_data() -> pl.DataFrame:
    """
    Loads the jobpost data from star data on the server
    """
    # get username
    username = os.popen("whoami").read().strip()  # noqa: S607 S605

    df = (
        pl.read_parquet(f"/home/{username}@PROD.SITAD.DK/code/jobads/src/dgp/textdata/output/jobads_clean.parquet")
        .select(pl.col("ann_id").alias("id"), pl.col("annonce_tekst").alias("text"))
        .filter(
            pl.col("text").is_not_null()  # a few obs have missing text but non-missing heading and rubrik
        )
    )
    return df


def df_to_tuple(df: pl.DataFrame) -> list[tuple[str, dict]]:
    """
    Convert a DataFrame to a list of (text, context) tuples.

    Args:
        df (pl.DataFrame): DataFrame with at least a 'text' column.

    Returns:
        list[tuple[str, dict]]: List of tuples with text and associated context (other columns).
    """
    text_col = df.select(pl.col("text")).to_series().to_list()
    df_dict = df.select(pl.all().exclude("text")).to_dicts()
    texts_tuple = [(text, context) for text, context in zip(text_col, df_dict)]
    return texts_tuple


def rename_jobcenter_obs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename rows with id 'Virksomheden har valgt at rekruttere via jobcentret' to unique IDs.

    Args:
        df (pl.DataFrame): Input DataFrame with an 'id' column.

    Returns:
        pl.DataFrame: DataFrame with unique IDs for jobcenter posts.
    """
    # Add a unique suffix to each duplicate "jobcenter" id using cumcount
    df = df.with_columns(
        pl.when(pl.col("id") == "Virksomheden har valgt at rekruttere via jobcentret")
        .then("jobcenter_" + (pl.cum_count("id").over("id") + 1).cast(pl.String))
        .otherwise(pl.col("id"))
        .alias("id")
    )
    return df


def load_excel(file_path: Path, sheet_name: str = "Sheet1") -> pl.DataFrame:
    """
    Load data from an Excel file and return it as a DataFrame.

    Args:
        file_path (Path): Path to the Excel file.
        sheet_name (str, optional): Name of the sheet to load. Defaults to "Sheet1".

    Returns:
        pl.DataFrame: DataFrame representing the rows in the Excel file.
    """
    df = pl.read_excel(
        file_path, sheet_name=sheet_name, columns=["ID", "Text"], schema_overrides={"ID": pl.String, "Text": pl.String}
    ).rename({"ID": "id", "Text": "text"})
    return df


def detect_language(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect the language of the texts using Lingua and add a 'language' column.

    Args:
        df (pl.DataFrame): DataFrame with a 'text' column.

    Returns:
        pl.DataFrame: DataFrame with an added 'language' column.
    """
    detector = LanguageDetectorBuilder.from_all_languages().build()
    languages = []
    texts = df.select(pl.col("text")).to_series().to_list()

    # We could possibly speed this up by setting the languages to detect
    outputs = detector.detect_languages_in_parallel_of(texts)
    for output in outputs:
        if output is not None:
            if hasattr(output, "iso_code_639_3"):
                languages.append(output.iso_code_639_3.name)
            else:
                raise UnsupportedLinguaOutput(str(output))
        else:
            languages.append("unknown")

    df = df.with_columns(pl.Series("language", languages))
    return df


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the DataFrame by renaming jobcenter posts and filtering for Danish language.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: Cleaned DataFrame.
    """
    # rename job center posts
    df = rename_jobcenter_obs(df)

    # remove non-danish posts
    df = df.filter(pl.col("language") == "DAN")

    return df


def register_extensions(extensions: tuple = ("id", "language", "clean_tokens")) -> None:
    """
    Register custom spaCy Doc extensions if not already set.

    Args:
        extensions (tuple): Tuple of extension names to register.
    """
    for extension in extensions:
        if not Doc.has_extension(extension):
            Doc.set_extension(extension, default=None)


def preprocess_texts(df: pl.DataFrame, par: DictConfig) -> DocBin:
    """
    Preprocess texts using spaCy: tokenization, lemmatization, stopword removal, and lowercasing.

    Args:
        df (pl.DataFrame): DataFrame with texts and context.
        par (DictConfig): Configuration parameters.

    Returns:
        DocBin: spaCy DocBin containing processed documents.
    """
    doc_bin = DocBin(store_user_data=True)

    # Load the spaCy language model
    nlp = spacy.load(par.preprocessing.model, enable=par.preprocessing.pipeline)

    # Set stop words
    if par.preprocessing.keep_negations is not None:
        negation_words = {"ikke", "nej", "ingen", "intet", "aldrig"}
        nlp.Defaults.stop_words -= negation_words

    # Register the extension for id if not already set
    register_extensions()

    # Extract texts and IDs from the DataFrame
    texts_tuple = df_to_tuple(df)

    for doc, context in tqdm(
        nlp.pipe(texts_tuple, as_tuples=True, batch_size=par.settings.batch_size, n_process=par.settings.threads),
        total=len(texts_tuple),
        desc="Preprocessing texts",
    ):
        doc._.id = context["id"]
        doc._.language = context["language"]
        doc._.clean_tokens = get_clean_tokens(doc)

        doc_bin.add(doc)

    return doc_bin


def corpus_unpack(corpus: DocBin) -> Generator[Doc, None, None]:
    """
    Unpack a spaCy DocBin corpus and yield Doc objects.

    Args:
        corpus (DocBin): The preprocessed corpus.

    Yields:
        Doc: spaCy Doc object.
    """
    nlp = spacy.blank("da")
    register_extensions()

    yield from corpus.get_docs(nlp.vocab)


def get_clean_tokens(doc: Doc) -> list[str]:
    """
    Extract cleaned tokens from a spaCy Doc (alphabetic, non-stopword, lemmatized, lowercased).

    Args:
        doc (Doc): A spaCy Doc object.

    Returns:
        list[str]: List of cleaned tokens.
    """
    return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]


def _build_tdm(   # type: ignore[no-any-unimported]
    texts: list, tdm_cell: str = "binary", ngram: int = 1, min_df: Optional[int | float] = None   # type: ignore[no-any-unimported]
) -> tuple[CountVectorizer | TfidfVectorizer, list[str]]:  # type: ignore[no-any-unimported]
    """
    Build a Term-Document Matrix (TDM) using sklearn and return a sparse matrix.

    Args:
        texts (list): List of tokenized texts (as strings).
        ids (list): List of document IDs.
        tdm_cell (str): Type of cell value ('binary', 'tf', or 'tfidf').
        ngram_range (tuple): Ngram range, e.g. (1, 2) for unigrams and bigrams.
        min_df (int | float, optional): Minimum document frequency for terms.

    Returns:
        tdm: sparse matrix
        vocab: list of terms
    """
    # Build the Term-Document
    tdm_args = {"token_pattern": "[^;]+", "ngram_range": (ngram, ngram), "min_df": min_df}

    if tdm_cell == "binary":
        # token_pattern is a regex for tokenization, not a password
        vectorizer = CountVectorizer(**tdm_args, binary=True)
    elif tdm_cell == "tf":
        vectorizer = CountVectorizer(**tdm_args)
    elif tdm_cell == "tfidf":
        vectorizer = TfidfVectorizer(**tdm_args)
    else:
        raise ValueError()

    tdm = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out().tolist()
    return tdm, vocab


def build_tdm(corpus: DocBin, par: DictConfig) -> tuple:
    """
    Build a Term-Document Matrix (TDM) from a spaCy DocBin corpus using a single vectorizer for all ngrams.

    Args:
        corpus (DocBin): Preprocessed spaCy DocBin.
        par (DictConfig): Configuration parameters.

    Returns:
        X: sparse matrix
        vocab: list of terms
        ids: list of document IDs
    """
    # Prepare documents and IDs
    texts = []
    ids = []
    for doc in tqdm(
        corpus_unpack(corpus),
        total=corpus.__len__(),
        desc="Unpacking corpus for TDM",
    ):
        ids.append(doc._.id)
        tokens = doc._.clean_tokens
        texts.append(";".join(tokens))

    # Use a single vectorizer for all ngrams
    tdm_list = []
    vocab = []
    for i, n in enumerate(range(1, par.tdm.ngram_n + 1)):
        # Build the Term-Document Matrix
        tdm, _vocab = _build_tdm(texts, tdm_cell=par.tdm.tdm_cell, ngram=n, min_df=par.tdm.min_df[i])
        # Append the sparse matrix and vocabulary
        tdm_list.append(tdm)
        vocab += _vocab

    # stack the sparse matrices together
    tdm = ss.hstack(tdm_list, format="csr")

    return tdm, vocab, ids


def export_tdm(tdm: pl.DataFrame, output_file: Path) -> None:
    """
    Export the Term-Document Matrix (TDM) to a Parquet file.

    Args:
        tdm (pl.DataFrame): The Term-Document Matrix.
        output_file (Path): Path to the output Parquet file.
    """
    tdm.write_parquet(output_file)


def export_corpus(corpus: DocBin, output_file: Path) -> None:
    """
    Export the preprocessed corpus to a .spacy binary file.

    Args:
        corpus (DocBin): The preprocessed corpus.
        output_file (Path): Path to the output file.
    """
    corpus.to_disk(output_file)


def export_corpus_split(corpus: DocBin, output_dir: Path, par: DictConfig) -> None:
    """
    Split a large DocBin into smaller chunks and save each as a separate .spacy file in a directory.
    Args:
        corpus (DocBin): The preprocessed corpus.
        output_dir (Path): Directory to save .spacy files.
        chunk_size (int): Number of docs per file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = list(corpus.get_docs(spacy.blank("da").vocab))
    n_chunks = math.ceil(len(docs) / par.settings.chunk_size)
    for i in range(n_chunks):
        chunk_bin = DocBin(store_user_data=True)
        for doc in docs[i * par.settings.chunk_size : (i + 1) * par.settings.chunk_size]:
            chunk_bin.add(doc)
        chunk_file = output_dir / f"corpus_{i + 1}.spacy"
        chunk_bin.to_disk(chunk_file)


def export_tdm_sparse(tdm: ss.csr_matrix, output_file: Path) -> None:  # type: ignore[no-any-unimported]
    """
    Export the sparse Term-Document Matrix (TDM) to a .npz file.
    Args:
        tdm (csr_matrix): Sparse matrix representing the Term-Document Matrix.
        output_file (Path): Path to the output file.
    """
    ss.save_npz(str(output_file), tdm)


def export_tdm_info(vocab: list[str], ids: list[str], output_file: Path) -> None:
    """
    Export the vocabulary list to a JSON file.
    Args:
        vocab (list[str]): List of terms in the vocabulary.
        ids (list[str]): List of document IDs.
        output_file (Path): Path to the output file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "ids": ids}, f)


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    params_path = project_root / "params.yaml"
    data_dir = project_root / "data"

    file_path = data_dir / "Jobnet.xlsx"
    corpus_file = data_dir / "corpus.spacy"
    corpus_dir = data_dir / "corpus_split"
    tdm_file = data_dir / "tdm.npz"
    tdm_info_file = data_dir / "tdm_info.json"

    # Load parameters
    par = OmegaConf.load(params_path).prepare

    # Process the data
    texts = load_data(file_path, par)[: par.settings.nobs]
    texts = detect_language(texts)
    texts = clean_data(texts)
    preprocessed_corpus = preprocess_texts(texts, par)
    tdm, vocab, ids = build_tdm(preprocessed_corpus, par)
    export_corpus_split(preprocessed_corpus, corpus_dir, par)
    print(f"Preprocessed corpus exported to {corpus_dir}")

    export_tdm_sparse(tdm, tdm_file)
    export_tdm_info(vocab, ids, tdm_info_file)
    print(f"Term-Document Matrix exported to {tdm_file}")
    print(f"TDM info exported to {tdm_info_file}")
