import os
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment

DIR = '/Users/zosia/Desktop/Test1/'
EXTENSION = ".ogg"
OUTPUTDIR = '/Users/zosia/Desktop/Test2/'
files = os.listdir(DIR)

def convert(file):
    filepath = os.path.join(DIR, file)
    outputpath = os.path.join(OUTPUTDIR, file)
    try:
        sound = AudioSegment.from_file(filepath)
        sound = sound.set_frame_rate(32000)
        sound.export(outputpath, format="ogg")
    except Exception as error:
        print("ERROR")


with ThreadPoolExecutor() as executor:
    results = executor.map(convert, files)
