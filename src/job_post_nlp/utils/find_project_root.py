import pathlib


class ProjectRootNotFoundError(FileNotFoundError):
    def __init__(self) -> None:
        super().__init__("Project root not found. No markers found in any parent directories.")


class InvalidStartPathTypeError(TypeError):
    def __init__(self, start_path: str | pathlib.Path) -> None:
        super().__init__(f"start_path must be a string or Path object, not {type(start_path)}")


class StartPathResolutionError(FileNotFoundError):
    def __init__(self, start_path: str | pathlib.Path) -> None:
        super().__init__(f"The starting path '{start_path}' does not exist or could not be resolved.")


def find_project_root(start_path: str | pathlib.Path, markers: list[str] | None = None) -> pathlib.Path:
    """
    Finds the project root directory by searching upwards from a starting path
    for specific marker files or directories.

    Args:
        start_path: The path to a file or directory within the project.
                    It's recommended to pass `__file__` from the calling script
                    to ensure the search starts relative to that script's location.
        markers: A list of filenames or directory names that indicate the
                 project root. If None, uses a default list:
                 ['.git', 'pyproject.toml', 'setup.py', '.project_root',
                  'requirements.txt', 'manage.py'].

    Returns:
        A pathlib.Path object representing the absolute path to the project
        root directory if found, otherwise None.

    Raises:
        FileNotFoundError: If the provided start_path does not exist.
        TypeError: If start_path is not a string or Path object.
    """
    if markers is None:
        # Common markers for various project types
        markers = [
            ".git",  # Git repository root
            "pyproject.toml",  # Standard Python project config
            "setup.py",  # Older Python project config
            ".project_root",  # Custom marker file
            "requirements.txt",  # Common dependency file often at root
            "manage.py",  # Django project root marker
            # Add other markers relevant to your projects if needed
        ]

    # Ensure start_path is an absolute Path object and exists
    if not isinstance(start_path, str | pathlib.Path):
        raise InvalidStartPathTypeError(start_path)
    try:
        # Use resolve() to get the absolute path and resolve any symlinks
        search_path = pathlib.Path(start_path).resolve(strict=True)
    except FileNotFoundError as err:
        raise StartPathResolutionError(start_path) from err

    # If start_path is a file, begin the search from its parent directory
    current_dir = search_path.parent if search_path.is_file() else search_path

    # Traverse upwards looking for markers
    while True:
        # Check if any marker exists in the current directory
        for marker in markers:
            if (current_dir / marker).exists():
                return current_dir  # Found the root

        # Move up to the parent directory
        parent_dir = current_dir.parent
        # Check if we have reached the filesystem root
        if parent_dir == current_dir:
            # This happens when current_dir is the root directory (e.g., '/')
            # or if permission errors prevent accessing higher directories.
            raise ProjectRootNotFoundError()

        current_dir = parent_dir
