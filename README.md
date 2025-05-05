# job-post-nlp


This is the code for the project "job-post-nlp", which is a Natural Language Processing (NLP) project focused on analyzing job postings. The project aims to extract insights from job descriptions, such as required skills, qualifications, and other relevant information.

Authors: Asker Christensen, Adam Hallengreen

- **Github repository**: <https://github.com/AdamHallengreen/job-post-nlp/>

## Running the project locally
To run this project locally, follow these steps:

### 1. Install UV for Python package management

First, install UV for python package management if you haven't already. If you are using Windows the recommended way is to use the PowerShell command below:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you are using macOS or Linux, the recommended way is to use the following command in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you don't already have Python installed, you can do this using UV by running:

```bash
uv install python 3.12
```

For alternative installation methods, see the [UV documentation](https://astral.sh/uv/installation/).

> **Note**: UV is a Python package management tool that simplifies the process of creating and managing virtual environments, installing dependencies, and running Python projects. It is designed to be fast, lightweight, and easy to use. To read more about UV, see the [UV documentation](https://astral.sh/uv/).

### 2. Download the repository
Clone the repository to your local machine using git. Locate the directory where you want to clone the repository and run:

```bash
git clone https://github.com/AdamHallengreen/job-post-nlp.git
```
### 3. Run the project
Navigate to the cloned repository directory in your terminal. Then run the following command to run the entire pipeline:

```bash
uv run dvc repro
```

This command will setup a virtual environment, install the required dependencies, and execute the DVC pipeline defined in the `dvc.yaml` file, which includes all the stages of your project.

> **Note**: DVC (Data Version Control) is a tool that helps manage machine learning projects by tracking data, code, and experiments. It allows you to define pipelines, version datasets, and share results with collaborators. The `dvc.yaml` file defines the stages of the pipeline, including data preprocessing, model training, and evaluation. Read more about DVC in the [DVC documentation](https://dvc.org/doc).

### 4. Inspect the results
After running the pipeline, you can inspect the results in the output directory. A report is also generated to summarize the key results.

## Development setup
To set up the development environment for this project, follow these steps:

### 1. Create a virtual environment

Install UV for Python package management if you haven't already, as described in the previous section. Navigate to the cloned repository directory in your terminal and run the following command to create a virtual environment:

```bash
uv sync
```
This command will create a virtual environment and install all the required dependencies specified in the `pyproject.toml` file. If you already ran `uv run dvc repro` a virtual environment has already been created.

> **Note**: If you are using VS Code, you can select the virtual environment as your interpreter by pressing `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS) and typing "Python: Select Interpreter". Then choose the interpreter that corresponds to the virtual environment created by UV.

### 2. Install pre-commit hooks
To ensure code quality and consistency, this project uses pre-commit hooks. To install them, run the following command:

```bash
uv run pre-commit install
```

This command will set up the pre-commit hooks defined in the `.pre-commit-config.yaml` file. These hooks will automatically run checks on your code before you commit any changes.

In addition to some native pre-commit hooks, this project also uses custom [Ruff pre-commit hooks](https://docs.astral.sh/ruff/integrations/#gitlab-cicd) for linting and formatting.

> **Note**: Ruff is a fast Python linter and formatter that can help you maintain code quality. Read more about Ruff in the [Ruff documentation](https://docs.astral.sh/ruff/).

Run the pre-commit hooks manually on all files in the repository to finalize the installation and ensure that your code is formatted correctly. To do this, run the following command:

```bash
uv run pre-commit run -a
```

### 3. Commit the changes

When you are done making changes to the code, commit your changes using your preferred Git management tool. I suggest using the command line as it is the most straightforward way to ensure that the pre-commit hooks are run before committing. Using the command line, you can commit your changes with the following commands:

```bash
git add .
git commit -m "Your commit message here"
```

Pre-commit hooks will automatically run when you commit your changes. If any of the hooks fail, you will need to address the issues before the commit can be completed. Some hooks automatically fix issues, while others will require you to manually address the problems.

> **Note**: If you are using VS Code I suggest installing the [ruff extension](https://marketplace.visualstudio.com/items?itemName=astral.rust) to help you identify and fix issues in your code as you work. The extension will automatically run Ruff on your code and provide suggestions for fixes.

When you pass the pre-commit hooks, you will see a message indicating that the commit was successful.

### 4. Push your changes
Once you have commited your code and passed the pre-commit hooks, you can push your changes to the remote repository. The remote main branch is protected, so you will need to create a development branch and open a pull request to merge your changes.

If you have been working on the main branch so far, save your work to a development branch first, and then push to the remote. Use your preferred Git management tool or the command line. Using the command line, this can be done with the following commands:

```bash
git checkout -b my-feature-branch
git push origin my-feature-branch
```

### 5. Merge into the main branch
To merge your changes into the main branch, you will need to create a pull request. Open [the remote repository on Github](https://github.com/AdamHallengreen/job-post-nlp) and create a pull request from your feature branch to the main branch. Navigate to the "Pull requests" tab and click on the "New pull request" button. Select your feature branch as the source branch and the main branch as the target branch.

When you create the pull request, GitHub will automatically run the CI/CD pipeline defined in the `.github/workflows/main.yml` file. This pipeline will run the pre-commit hooks, tests, type checks (using [mypy](https://mypy.readthedocs.io/en/stable/)), and create a report of the results.

> **Note**: Mypy is a static type checker for Python. It helps identify type errors in your code by verifying variable types, function arguments, and return values against expected types. For more details, refer to the [mypy documentation](https://mypy.readthedocs.io/en/stable/).

If any of these checks fail, you will need to address the issues before merging your changes.



You are now ready to contribute to the project!


---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
