import os
import librosa
import json

from tqdm import tqdm

# test
TST_TXT_PATH = '/media/sf_DATA/ASVSpoof/ASVspoof 2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
TST_DIR_PATH = 'test'
TST_JSON_PATH = 'test.json'

# train
TRN_TXT_PATH = '/media/sf_DATA/ASVSpoof/ASVspoof 2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
TRN_DIR_PATH = '/media/sf_DATA/ASVSpoof/ASVspoof 2019/LA/ASVspoof2019_LA_train/flac'
TRN_JSON_PATH = 'train_data.json'

# dev
DEV_TXT_PATH = '/media/sf_DATA/ASVSpoof/ASVspoof 2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
DEV_DIR_PATH = '/media/sf_DATA/ASVSpoof/ASVspoof 2019/LA/ASVspoof2019_LA_dev/flac'
DEV_JSON_PATH = 'dev_data.json'

# eval
EVL_TXT_PATH = '/media/sf_DATA/ASVSpoof/ASVspoof 2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
EVL_DIR_PATH = '/media/sf_DATA/ASVSpoof/ASVspoof 2019/LA/ASVspoof2019_LA_eval/flac'
EVL_JSON_PATH = 'eval_data.json'

# create dictionary
data = {
    "filename": [],
    "id": [],
    "mfcc": [],
    "label": [],
    "class": []
}

# modes
modes = ['DEV', 'TRN', 'EVL']

# specify number of MFCCs being created
n_mfcc = 13


def processData(TXT_PATH, JSON_PATH, DIR_PATH):
    # open txt document
    with open(TXT_PATH, 'r') as txtfile:

        # open json
        with open(JSON_PATH, 'w') as fp:

            # calculate number of files
            num_files = len(os.listdir(DIR_PATH))

            # after each line
            for line in tqdm(txtfile, total=num_files, position=0, leave=True):

                # check folder for file
                for i, flacfile in enumerate(os.listdir(DIR_PATH)):

                    # get the name of the audio file from the TXT file
                    filename = flacfile[0:12]

                    # if file is found in folder
                    if filename in line:

                        # add filename to dictionary
                        data["filename"].append(filename)

                        data["id"].append(line[3:7])

                        # get file path
                        FILE_PATH = os.path.join(DIR_PATH, flacfile)

                        # load audio file
                        signal, sr = librosa.load(FILE_PATH)
                        duration = librosa.get_duration(signal)

                        # calculate MFCC
                        mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc)
                        mfcc = mfcc.T

                        # append data to dictionary
                        data["mfcc"].append(mfcc.tolist())

                        # check label and add it to dictionary
                        if 'bonafide' in line:
                            label = 0
                            data["label"].append(label)
                            data["class"].append("bonafide")
                            continue
                        else:
                            label = 1
                            data["label"].append(label)
                            labelclass = line[23:26]
                            data["class"].append(labelclass)
                            continue

            # save to JSON file
            json.dump(data, fp, indent=4)

    # clean up
    data.clear()
    print("File ", JSON_PATH, " created")

processData(DEV_TXT_PATH, DEV_JSON_PATH, DEV_DIR_PATH)
processData(TRN_TXT_PATH, TRN_JSON_PATH, TRN_DIR_PATH)
processData(EVL_TXT_PATH, EVL_JSON_PATH, EVL_DIR_PATH)