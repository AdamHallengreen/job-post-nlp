##only use these line when running in an interactive window
def try_inter() -> None:
    """
    Runs the autoreload magic command in IPython if running in an interactive window.
    This allows for automatic reloading of modules when they are modified, which is useful for
    development and testing purposes.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
    except ImportError:
        pass
