import os
from pydub import AudioSegment

DIR = '/media/jacek/E753-A120/recordings_30/'
EXTENSION = ".ogg"

# #convert
# for path, subdirs, files in os.walk(DIR):
#     for name in files:
#         filepath = os.path.join(path, name)
#         try:
#             sound = AudioSegment.from_file(filepath)
#             sound = sound.set_frame_rate(32000)
#             sound.export(filepath, format="ogg")
#         except Exception as error:
#             print(name)
#
# #change extension
# for path, subdirs, files in os.walk(DIR):
#     for name in files:
#         filepath_new = os.path.join(path, name.split('.')[0] + '.ogg')
#         filepath_old = os.path.join(path, name)
#         os.rename(filepath_old, filepath_new)
DIRS = ['Sturnus vulgaris', 'Pica pica', 'Passer domesticus', 'Grus grus', 'Garrulus glandarius', 'Corvus cornix']
DIRS = ['Carduelis carduelis', 'Asio otus']
SOURCE = '/media/jacek/E753-A120/recordings_30/'

for i in DIRS:
    DIR = SOURCE + i + "/"
    files = os.listdir(DIR)
    for file in files:
        try:
            print(f"{DIR}{file}")
            sound = AudioSegment.from_file(f"{DIR}{file}")
            sound = sound.set_frame_rate(32000)
            sound.export(f"{DIR}{file}", format="ogg")
        except Exception as error:
            print("An exception occurred:", error)
            print(file)
