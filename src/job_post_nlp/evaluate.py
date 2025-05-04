import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dvclive import Live

from job_post_nlp.utils.find_project_root import find_project_root


class InvalidInputFileError(Exception):
    def __init__(self) -> None:
        super().__init__("Input file must contain a list of lists.")


def load_most_common_words(input_file: str | Path) -> list:
    """
    Load the most common words from a JSON file.

    Args:
        input_file (str | Path): Path to the JSON file containing the most common words.

    Returns:
        list: A list of tuples containing words and their counts.
    """
    with open(input_file, encoding="utf-8") as f:
        words = json.load(f)
    if not isinstance(words, list) or not all(isinstance(word, list) and len(word) == 2 for word in words):
        raise InvalidInputFileError()
    return words


def plot_most_common_words(common_words: list) -> plt.Figure:
    """
    Create a bar plot of the most common words and save it as an image.

    Args:
        common_words (list): A list of tuples containing words and their counts.
        output_image (str): Path to the output image file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes
    words, counts = zip(*common_words)  # Unpack words and counts
    ax.bar(words, counts, color="skyblue")  # Create bar plot
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    ax.set_title("Most Common Words")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Define file paths
    project_root = find_project_root(__file__)
    input_file = Path(project_root) / "data" / "most_common_words.json"
    output_path = Path(project_root) / "output"
    metrics = Path(project_root) / "metrics.yaml"

    # Load the most common words
    common_words = load_most_common_words(input_file)

    # Log metrics using DVCLive
    with Live(dir=str(output_path), cache_images=True) as live:
        for i, (word, _) in enumerate(common_words):
            live.log_metric(f"Top {i + 1}", word, plot=False)

        # transform list of tuples to pd.DataFrame
        common_words_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
        live.log_plot(
            "Most Common Words",
            common_words_df,
            x="Frequency",
            y="Word",
            template="bar_horizontal",
            title="Most Common Words Bar Plot",
        )

        fig = plot_most_common_words(common_words)
        live.log_image("most_common_words.png", fig)
