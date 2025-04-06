#
# update_nlpia2_data.sh
import shutil
import os

from nlpia2.constants import MANUSCRIPT_DIR as FROM_PATH
from nlpia2.constants import SRC_DATA_DIR, DATA_DIR


def copytree_recursively(from_path=SRC_DATA_DIR, to_path=DATA_DIR, force=False):
    """ Use this ONLY ON SUBDIRECTORIES OF nlpia2-data/ , e.g. nlpia2-data/manuscript """
    if force and os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


# def copytree(from_path=SRC_DATA_DIR, to_path=DATA_DIR, force=False):
#     """ Use this ONLY ON SUBDIRECTORIES OF nlpia2-data/ , e.g. nlpia2-data/manuscript """
#     if force and os.path.exists(to_path):
#         shutil.rmtree(to_path)
#     shutil.copytree(from_path, to_path)
