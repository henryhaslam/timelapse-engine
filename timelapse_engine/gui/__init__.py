"""Desktop GUI (PySide6). Start it with ``python -m timelapse_engine.gui``."""


def main(argv=None) -> int:
    from .app import main as _main

    return _main(argv)
