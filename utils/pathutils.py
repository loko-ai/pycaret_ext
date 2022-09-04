from pathlib import Path


def find_path(name):
    p = Path(".").resolve()
    while str(p) != p.root:
        temp = p / name
        if temp.exists():
            return temp
        else:
            p = p.parent

    raise FileNotFoundError(name)
