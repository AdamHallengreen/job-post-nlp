import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from job_post_nlp.utils.find_project_root import find_project_root


def load_tokens(input_file: str):
    """
    Load tokenized texts from a JSON file.

    Args:
        input_file (str): Path to the JSON file containing tokenized texts.

    Returns:
        list: A list of tokenized texts.
    """
    with open(input_file, encoding="utf-8") as f:
        tokens = json.load(f)
    return tokens


def get_most_common_words(tokens: list, params):
    """
    Get the most common words from a list of tokenized texts.

    Args:
        tokens (list): A list of tokenized texts.
        top_n (int): Number of most common words to return.

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


def export_most_common_words(common_words: list, output_file: str):
    """
    Export the most common words to a file in JSON format.

    Args:
        common_words (list): A list of tuples containing words and their counts.
        output_file (str): Path to the output file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(common_words, f, ensure_ascii=False, indent=4)


def plot_most_common_words(common_words: list, output_image: str):
    """
    Create a bar plot of the most common words and save it as an image.

    Args:
        common_words (list): A list of tuples containing words and their counts.
        output_image (str): Path to the output image file.
    """
    words, counts = zip(*common_words)  # Unpack words and counts
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color="skyblue")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Most Common Words")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image)
    plt.close()


if __name__ == "__main__":
    # Define file paths
    project_root = find_project_root(__file__)
    params_path = Path(project_root) / "params.yaml"
    input_file = Path(project_root) / "data" / "tokenized_texts.json"
    output_file = Path(project_root) / "outputs" / "most_common_words.json"
    output_image = Path(project_root) / "outputs" / "most_common_words_bar_plot.png"

    # Load parameters
    with open(params_path) as file:
        params = yaml.safe_load(file)["most_common"]

    # Process
    tokens = load_tokens(input_file)
    common_words = get_most_common_words(tokens, params)
    export_most_common_words(common_words, output_file)
    plot_most_common_words(common_words, output_image)

    print(f"Most common words exported to {output_file}")
    print(f"Bar plot of most common words exported to {output_image}")
