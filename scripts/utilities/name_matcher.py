"""
Given two dirpaths, this script deletes all files in dir2 that do not have a corresponding file in dir1.
"""
import os
from simple_colors import red


def sync_dirs(dir1, dir2, delete_matched=False, match_extension=True):
    """
    Deletes files in dir2 based on their presence in dir1, according to the delete_matched flag,
    and considers file extensions based on the match_extension flag.

    Parameters:
    - dir1: Path to the source directory.
    - dir2: Path to the target directory where files will be deleted based on the delete_matched flag.
    - delete_matched: If True, deletes files in dir2 that have a corresponding file in dir1. If False, deletes files in dir2 that do not have a corresponding file in dir1.
    - match_extension: If True, considers file extensions in the match. If False, ignores file extensions.
    """
    # Adjust how files are listed based on whether extensions are considered
    if match_extension:
        files_in_dir1 = set(os.listdir(dir1))
        files_in_dir2 = set(os.listdir(dir2))
    else:
        files_in_dir1 = set(os.path.splitext(file)[0] for file in os.listdir(dir1))
        files_in_dir2 = set(os.path.splitext(file)[0] for file in os.listdir(dir2))

    if delete_matched:
        # Determine files in dir2 present in dir1
        files_to_delete = files_in_dir2.intersection(files_in_dir1)
    else:
        # Determine files in dir2 not present in dir1
        files_to_delete = files_in_dir2 - files_in_dir1

    # For no extension match, map names back to actual filenames in dir2 to delete
    if not match_extension:
        actual_files_in_dir2 = {os.path.splitext(file)[0]: file for file in os.listdir(dir2)}
        files_to_delete = {actual_files_in_dir2[file] for file in files_to_delete if file in actual_files_in_dir2}

    count = len(files_to_delete)
    print(red(f"Deleting {count} files from {dir2}...", "bold"))

    # Delete specified files from dir2
    for file in files_to_delete:
        os.remove(os.path.join(dir2, file))

    print("Sync complete.")


if __name__ == "__main__":
    dirpath1 = "/home/dynamo/AMRL_Research/repos/nspl/evals_data_parking/utcustom/train/images"
    dirpath2 = "/home/dynamo/AMRL_Research/repos/nspl/evals_data_parking/utcustom/train/pcs"
    sync_dirs(dirpath1, dirpath2, delete_matched=False, match_extension=False)
