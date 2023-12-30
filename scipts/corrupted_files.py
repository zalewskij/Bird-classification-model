"""
While moving files with convert_multithreading.py the files that are not opened correctly (missing recordings in xeno canto)
throw exception are not moved.
Hence, we can detect corrupted files comparing folder with downloads and folder after conversion
"""
from pathlib import Path
import os

DOWNLOAD_DIRECTORY = Path("/media/jacek/E753-A120/xeno-canto-splitted")
CONVERTED_DIRECTORY = Path("/media/jacek/E753-A120/xeno-canto-splitted-converted")


#list all downloaded files
downloaded_files = []
for path, subdirs, files in os.walk(DOWNLOAD_DIRECTORY):
    for name in files:
        downloaded_files.append(os.path.join(name))

#list all converted files
converted_files = []
for path, subdirs, files in os.walk(CONVERTED_DIRECTORY):
    for name in files:
        converted_files.append(os.path.join(name))

#print number of files in each directory
print("Downloaded files:", len(downloaded_files))
print("Converted files:", len(converted_files))

#difference
differences = list(set(converted_files).symmetric_difference(set(downloaded_files)))
for id in differences:
    print(id, end=', ')

