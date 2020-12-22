import os, fnmatch

def get_files(directory, extension="*"):
    """
    walk through the given directory and return all files has the given extension.
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, "*."+extension):
            files.append(os.path.join(root, filename))
    return files