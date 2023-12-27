import os
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment

DIR = '/media/jacek/E753-A120/xeno-canto-splitted'
EXTENSION = ".ogg"
OUTPUTDIR = '/media/jacek/E753-A120/xeno-canto-splitted-converted'
filepath_list = []

for path, subdirs, files in os.walk(DIR):
    for name in files:
        filepath_list.append(os.path.join(path, name))

filepath_list = filepath_list[250000:275000]
print(len(filepath_list))


def convert(filepath):
    output_filepath = filepath.replace('xeno-canto-splitted', 'xeno-canto-splitted-converted')
    try:
        sound = AudioSegment.from_file(filepath)
        sound = sound.set_frame_rate(32000)
        sound.export(output_filepath, format="ogg")
    except Exception as error:
        print("ERROR: ", filepath)


with ThreadPoolExecutor(max_workers = 4) as executor:
    results = executor.map(convert, filepath_list)
