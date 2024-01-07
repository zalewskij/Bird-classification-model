from birdclassification.training.preprocessing_pipeline import PreprocessingPipeline
import pandas as pd
from birdclassification.preprocessing.filtering import filter_recordings_287
from birdclassification.training.dataset import Recordings30
from birdclassification.training.cnn_training_torch.CNN_model import CNNNetwork
from birdclassification.training.training_utils import train_one_epoch
from birdclassification.training.validation_metrics import calculate_metric
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from pathlib import Path
from datetime import datetime
import sys
from time import time
from birdclassification.preprocessing.utils import oversample_dataframe, undersample_dataframe

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 123
BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
#RECORDINGS_DIR = Path('/mnt/d/recordings_30')
RECORDINGS_DIR = Path("/media/jacek/E753-A120/recordings_287")
NOISES_DIR = Path('/media/jacek/E753-A120/NotBirds')
WRITER_DIR = Path(__file__).resolve().parent / "logs"
MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "cnn_1.pt"
SAMPLE_RATE = 32000
BATCH_SIZE = 32
NUM_WORKERS = 8
LEARNING_RATE = 0.0001
EPOCHS = 20

MAPPING = {0: 'Acanthis cabaret', 1: 'Acanthis flammea', 2: 'Accipiter gentilis', 3: 'Accipiter nisus', 4: 'Acrocephalus arundinaceus',
           5: 'Acrocephalus dumetorum', 6: 'Acrocephalus paludicola', 7: 'Acrocephalus palustris', 8: 'Acrocephalus schoenobaenus',
           9: 'Acrocephalus scirpaceus', 10: 'Actitis hypoleucos', 11: 'Aegithalos caudatus', 12: 'Aegolius funereus',
           13: 'Aix galericulata', 14: 'Alauda arvensis', 15: 'Alca torda', 16: 'Alcedo atthis', 17: 'Alopochen aegyptiaca',
           18: 'Anas acuta', 19: 'Anas crecca', 20: 'Anas platyrhynchos', 21: 'Anser albifrons', 22: 'Anser anser',
           23: 'Anser brachyrhynchus', 24: 'Anser erythropus', 25: 'Anser fabalis', 26: 'Anser serrirostris', 27: 'Anthus campestris',
           28: 'Anthus cervinus', 29: 'Anthus petrosus', 30: 'Anthus pratensis', 31: 'Anthus spinoletta', 32: 'Anthus trivialis',
           33: 'Apus apus', 34: 'Aquila chrysaetos', 35: 'Ardea alba', 36: 'Ardea cinerea', 37: 'Ardea purpurea',
           38: 'Arenaria interpres', 39: 'Asio flammeus', 40: 'Asio otus', 41: 'Athene noctua', 42: 'Aythya ferina',
           43: 'Aythya fuligula', 44: 'Aythya marila', 45: 'Aythya nyroca', 46: 'Bombycilla garrulus',
           47: 'Botaurus stellaris', 48: 'Branta bernicla', 49: 'Branta canadensis', 50: 'Branta leucopsis',
           51: 'Branta ruficollis', 52: 'Bubo bubo', 53: 'Bucephala clangula', 54: 'Buteo buteo', 55: 'Buteo lagopus',
           56: 'Buteo rufinus', 57: 'Calcarius lapponicus', 58: 'Calidris alba', 59: 'Calidris alpina', 60: 'Calidris canutus',
           61: 'Calidris falcinellus', 62: 'Calidris ferruginea', 63: 'Calidris minuta', 64: 'Calidris pugnax', 65: 'Calidris temminckii',
           66: 'Caprimulgus europaeus', 67: 'Carduelis carduelis', 68: 'Carpodacus erythrinus', 69: 'Cepphus grylle', 70: 'Certhia brachydactyla',
           71: 'Certhia familiaris', 72: 'Charadrius dubius', 73: 'Charadrius hiaticula', 74: 'Charadrius morinellus', 75: 'Chlidonias hybrida',
           76: 'Chlidonias leucopterus', 77: 'Chlidonias niger', 78: 'Chloris chloris', 79: 'Chroicocephalus ridibundus', 80: 'Ciconia ciconia',
           81: 'Ciconia nigra', 82: 'Cinclus cinclus', 83: 'Circaetus gallicus', 84: 'Circus aeruginosus', 85: 'Circus cyaneus',
           86: 'Circus macrourus', 87: 'Circus pygargus', 88: 'Clanga clanga', 89: 'Clanga pomarina', 90: 'Clangula hyemalis',
           91: 'Coccothraustes coccothraustes', 92: 'Coloeus monedula', 93: 'Columba livia', 94: 'Columba oenas', 95: 'Columba palumbus',
           96: 'Coracias garrulus', 97: 'Corvus corax', 98: 'Corvus cornix', 99: 'Corvus corone', 100: 'Corvus frugilegus', 101: 'Coturnix coturnix',
           102: 'Crex crex', 103: 'Cuculus canorus', 104: 'Curruca communis', 105: 'Curruca curruca', 106: 'Curruca nisoria', 107: 'Cyanistes caeruleus',
           108: 'Cygnus columbianus', 109: 'Cygnus cygnus', 110: 'Cygnus olor', 111: 'Delichon urbicum', 112: 'Dendrocopos leucotos',
           113: 'Dendrocopos major', 114: 'Dendrocopos syriacus', 115: 'Dendrocoptes medius', 116: 'Dryobates minor', 117: 'Dryocopus martius',
           118: 'Egretta garzetta', 119: 'Emberiza calandra', 120: 'Emberiza citrinella', 121: 'Emberiza hortulana', 122: 'Emberiza schoeniclus',
           123: 'Eremophila alpestris', 124: 'Erithacus rubecula', 125: 'Falco columbarius', 126: 'Falco peregrinus', 127: 'Falco subbuteo',
           128: 'Falco tinnunculus', 129: 'Falco vespertinus', 130: 'Ficedula albicollis', 131: 'Ficedula hypoleuca', 132: 'Ficedula parva',
           133: 'Fringilla coelebs', 134: 'Fringilla montifringilla', 135: 'Fulica atra', 136: 'Galerida cristata', 137: 'Gallinago gallinago',
           138: 'Gallinago media', 139: 'Gallinula chloropus', 140: 'Garrulus glandarius', 141: 'Gavia arctica', 142: 'Gavia stellata',
           143: 'Glaucidium passerinum', 144: 'Grus grus', 145: 'Haematopus ostralegus', 146: 'Haliaeetus albicilla', 147: 'Himantopus himantopus',
           148: 'Hippolais icterina', 149: 'Hirundo rustica', 150: 'Hydrocoloeus minutus', 151: 'Hydroprogne caspia', 152: 'Ichthyaetus melanocephalus',
           153: 'Ixobrychus minutus', 154: 'Jynx torquilla', 155: 'Lanius collurio', 156: 'Lanius excubitor', 157: 'Larus argentatus',
           158: 'Larus cachinnans', 159: 'Larus canus', 160: 'Larus fuscus', 161: 'Larus marinus', 162: 'Larus michahellis', 163: 'Limosa lapponica',
           164: 'Limosa limosa', 165: 'Linaria cannabina', 166: 'Linaria flavirostris', 167: 'Locustella fluviatilis', 168: 'Locustella luscinioides',
           169: 'Locustella naevia', 170: 'Lophophanes cristatus', 171: 'Loxia curvirostra', 172: 'Lullula arborea', 173: 'Luscinia luscinia',
           174: 'Luscinia megarhynchos', 175: 'Luscinia svecica', 176: 'Lymnocryptes minimus', 177: 'Lyrurus tetrix', 178: 'Mareca penelope',
           179: 'Mareca strepera', 180: 'Melanitta fusca', 181: 'Melanitta nigra', 182: 'Mergellus albellus', 183: 'Mergus merganser',
           184: 'Mergus serrator', 185: 'Merops apiaster', 186: 'Milvus migrans', 187: 'Milvus milvus', 188: 'Motacilla alba',
           189: 'Motacilla cinerea', 190: 'Motacilla citreola', 191: 'Motacilla flava', 192: 'Muscicapa striata', 193: 'Netta rufina',
           194: 'Nucifraga caryocatactes', 195: 'Numenius arquata', 196: 'Numenius phaeopus', 197: 'Nycticorax nycticorax',
           198: 'Oenanthe oenanthe', 199: 'Oriolus oriolus', 200: 'Pandion haliaetus', 201: 'Panurus biarmicus', 202: 'Parus major',
           203: 'Passer domesticus', 204: 'Passer montanus', 205: 'Perdix perdix', 206: 'Periparus ater', 207: 'Pernis apivorus',
           208: 'Phalacrocorax carbo', 209: 'Phalaropus lobatus', 210: 'Phasianus colchicus', 211: 'Phoenicurus ochruros',
           212: 'Phoenicurus phoenicurus', 213: 'Phylloscopus collybita', 214: 'Phylloscopus inornatus', 215: 'Phylloscopus sibilatrix',
           216: 'Phylloscopus trochiloides', 217: 'Phylloscopus trochilus', 218: 'Pica pica', 219: 'Picoides tridactylus', 220: 'Picus canus',
           221: 'Picus viridis', 222: 'Plectrophenax nivalis', 223: 'Pluvialis apricaria', 224: 'Pluvialis squatarola', 225: 'Podiceps auritus',
           226: 'Podiceps cristatus', 227: 'Podiceps grisegena', 228: 'Podiceps nigricollis', 229: 'Poecile montanus', 230: 'Poecile palustris',
           231: 'Porzana porzana', 232: 'Prunella collaris', 233: 'Prunella modularis', 234: 'Pyrrhula pyrrhula', 235: 'Rallus aquaticus',
           236: 'Recurvirostra avosetta', 237: 'Regulus ignicapilla', 238: 'Regulus regulus', 239: 'Remiz pendulinus', 240: 'Riparia riparia',
           241: 'Rissa tridactyla', 242: 'Saxicola rubetra', 243: 'Saxicola rubicola', 244: 'Scolopax rusticola', 245: 'Serinus serinus',
           246: 'Sitta europaea', 247: 'Somateria mollissima', 248: 'Spatula clypeata', 249: 'Spatula querquedula', 250: 'Spinus spinus',
           251: 'Stercorarius parasiticus', 252: 'Sterna hirundo', 253: 'Sterna paradisaea', 254: 'Sternula albifrons', 255: 'Streptopelia decaocto',
           256: 'Streptopelia turtur', 257: 'Strix aluco', 258: 'Strix uralensis', 259: 'Sturnus vulgaris', 260: 'Sylvia atricapilla', 261: 'Sylvia borin',
           262: 'Tachybaptus ruficollis', 263: 'Tadorna ferruginea', 264: 'Tadorna tadorna', 265: 'Tetrao urogallus', 266: 'Tetrastes bonasia',
           267: 'Thalasseus sandvicensis', 268: 'Tichodroma muraria', 269: 'Tringa erythropus', 270: 'Tringa glareola', 271: 'Tringa nebularia',
           272: 'Tringa ochropus', 273: 'Tringa stagnatilis', 274: 'Tringa totanus', 275: 'Troglodytes troglodytes', 276: 'Turdus iliacus', 277: 'Turdus merula',
           278: 'Turdus philomelos', 279: 'Turdus pilaris', 280: 'Turdus torquatus', 281: 'Turdus viscivorus', 282: 'Tyto alba', 283: 'Upupa epops',
           284: 'Uria aalge', 285: 'Vanellus vanellus', 286: 'Zapornia parva'}

df = filter_recordings_287(BASE_PATH / "data" / "xeno_canto_recordings.csv", BASE_PATH / "data" / "bird-list-extended.csv")
noises_df = pd.read_csv(Path('../../../data/noises.csv'))
train_df, test_val_df = train_test_split(df, stratify=df['Latin name'], test_size=0.2, random_state = SEED)
val_df, test_df = train_test_split(test_val_df, stratify=test_val_df['Latin name'], test_size=0.5, random_state = SEED)

train_df = oversample_dataframe(train_df, minimum_number_of_samples=1000, mapping=MAPPING, seed=SEED)
train_df = undersample_dataframe(train_df, maximum_number_of_samples=3000, mapping=MAPPING, seed=SEED)
train_df = train_df.sample(frac=1.0, random_state=SEED)

train_ds = Recordings30(train_df, recording_dir=RECORDINGS_DIR, device = DEVICE, random_fragment=True)
val_ds = Recordings30(val_df, recording_dir=RECORDINGS_DIR, device = DEVICE)
test_ds = Recordings30(test_df, recording_dir=RECORDINGS_DIR, device = DEVICE)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

cnn = CNNNetwork().to(DEVICE)
summary(cnn, (1, 64, 251))
cnn.eval()
preprocessing_pipeline = PreprocessingPipeline(device=DEVICE, random_fragment=True, noises_dir=NOISES_DIR, noises_df=noises_df).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(),
                             lr=LEARNING_RATE)

writer = SummaryWriter(WRITER_DIR)
epoch_number = 0

best_vloss = sys.float_info.max

print("--------------------")
print(f"TIMESTAMP: {TIMESTAMP}")
print(f"DEVICE: {DEVICE}")
print(f"SEED: {SEED}")
print(f"BASE_PATH: {BASE_PATH}")
print(f"RECORDINGS_DIR: {RECORDINGS_DIR}")
print(f"NOISES_DIR: {NOISES_DIR}")
print(f"WRITER_DIR: {WRITER_DIR}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"SAMPLE_RATE: {SAMPLE_RATE}")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"NUM_WORKERS: {NUM_WORKERS}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"EPOCHS: {EPOCHS}")
print("--------------------")

start_time = time()

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    epoch_start_time = time()
    
    # Make sure gradient tracking is on, and do a pass over the data
    cnn.train(True)
    avg_loss = train_one_epoch(epoch_number, preprocessing_pipeline, writer, train_dl, optimizer, loss_fn, cnn, DEVICE, start_time)

    # Set the model to evaluation mode, disabling dropout and using population 
    # statistics for batch normalization.
    cnn.eval()
    running_vloss = 0.0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dl):
            vinputs, vlabels = vdata
            vinputs = preprocessing_pipeline(vinputs.to(DEVICE), use_augmentations=False)
            voutputs = cnn(vinputs)
            vloss = loss_fn(voutputs, vlabels.to(DEVICE))
            running_vloss += vloss
    
    avg_vloss = running_vloss / (i + 1)
    print("#############################################################")
    print("Epoch results:")
    print(f'Loss train {avg_loss} valid loss: {avg_vloss}')
    validation_precision_score = calculate_metric(cnn, val_dl, preprocessing_pipeline=preprocessing_pipeline, device=DEVICE, metric=lambda x, y: precision_score(x, y, average='macro'))
    print(f'Validation macro avarage precision: {validation_precision_score}')
    print(f'Epoch execution time {time() - epoch_start_time}')
    print("#############################################################\n\n")
    
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    
    
    writer.add_scalars('Macro_averaged_precision_score',
                    { 'Validation' : validation_precision_score},
                    epoch_number + 1)
    
    writer.flush()
    
    # Track best performance, and save the model's state
    best_vloss = avg_vloss
    model_path = f'model_{TIMESTAMP}_{epoch_number}'
    torch.save(cnn.state_dict(), MODEL_PATH.parent / model_path)
    
    epoch_number += 1

torch.save(cnn.state_dict(), MODEL_PATH)
