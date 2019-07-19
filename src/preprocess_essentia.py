import os
from essentia.standard import *
from joblib import Parallel, delayed
import pickle
import json
import config_file
import argparse
import numpy as np
from pathlib import Path

DEBUG = False

def compute_audio_repr(audio_file, audio_repr_file):

    # Compute audio representation
    if config['type'] == 'audioset': # for audioset features, use their implementation!
        import vggish_input
        audio_repr = vggish_input.wavfile_to_examples(audio_file)
        print(audio_repr.shape)

    else: # if is not audioset, use essentia!

        loader = essentia.standard.MonoLoader(filename=audio_file)
        audio = loader()

        w = Windowing(type='hann')
        spectrum = Spectrum() # FFT() would return the complex FFT, here we just want the magnitude spectrum
        mels = MelBands(numberBands=config['n_mels'], type='magnitude')
        logNorm = UnaryOperator(type='log')

        # mel_spectrogram = []
        log_mel_spectrogram = []

        for frame in FrameGenerator(audio, frameSize=config['n_fft'], hopSize=config['hop'], startFromZero=True):
            mel_frame = mels(spectrum(w(frame)))
            # mel_spectrogram.append(mel_frame)
            log_mel_spectrogram.append(logNorm(mel_frame))

        # transpose to have it in a better shape
        # we need to convert the list to an essentia.array first (== numpy.array of floats)
        audio_repr = essentia.array(log_mel_spectrogram)
        print(audio_repr.shape)
    
    # Compute length
    length = audio_repr.shape[0]

    # Write results:
    with open(audio_repr_file, "wb") as f:
        pickle.dump(audio_repr, f)  # audio_repr shape: NxM

    return length


def do_process(files, index):
    try:
        [id, audio_file, audio_repr_file] = files[index]
        if not os.path.exists(audio_repr_file[:audio_repr_file.rfind('/') + 1]):
            path = Path(audio_repr_file[:audio_repr_file.rfind('/') + 1])
            path.mkdir(parents=True, exist_ok=True)
        length = compute_audio_repr(audio_file, audio_repr_file)
        # index.tsv writing
        fw = open(config_file.DATA_FOLDER + config['audio_representation_folder'] + "index_" + str(config['machine_i'])
                  + ".tsv", "a")
        fw.write("%s\t%s\t%s\n" % (
                 id, audio_repr_file[len(config_file.DATA_FOLDER):], audio_file[len(config_file.DATA_FOLDER):]))
        fw.close()
        print(str(index) + '/' + str(len(files)) + ' Computed: %s' % audio_file)
    except Exception as e:
        ferrors = open(config_file.DATA_FOLDER + config['audio_representation_folder'] + "errors" + str(config['machine_i'])
                       + ".txt", "a")
        ferrors.write(audio_file + "\n")
        ferrors.write(str(e))
        ferrors.close()
        print('Error computing audio representation: ', audio_file)
        print(str(e))


def process_files(files):
    if DEBUG:
        print('WARNING: Parallelization is not used!')
        for index in range(0, len(files)):
            do_process(files, index)

    else:
        Parallel(n_jobs=config['num_processing_units'])(
            delayed(do_process)(files, index) for index in range(0, len(files)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('configurationID', help='ID of the configuration dictionary')
    args = parser.parse_args()
    config = config_file.config_preprocess[args.configurationID]

    config['audio_representation_folder'] = "audio_representation/%s__%s/" % (config['identifier'], config['type'])
    # set audio representations folder
    if not os.path.exists(config_file.DATA_FOLDER + config['audio_representation_folder']):
        os.makedirs(config_file.DATA_FOLDER + config['audio_representation_folder'])
    else:
        print("WARNING: already exists a folder with this name!"
              "\nThis is expected if you are splitting computations into different machines.."
              "\n..because all these machines are writing to this folder. Otherwise, check your config_file!")

    # list audios to process: according to 'index_file'
    files_to_convert = []
    f = open(config_file.DATA_FOLDER + config["index_file"])
    for line in f.readlines():
        id, audio = line.strip().split("\t")
        if config['convert_id']:
            audio_repr = id + ".pk"
        else:
            audio_repr = audio[:audio.rfind(".")] + ".pk"
        files_to_convert.append((id, config['audio_folder'] + audio,
                                 config_file.DATA_FOLDER + config['audio_representation_folder'] + audio_repr))

    # compute audio representation: waveform or spectrogram
    if config['machine_i'] == config['n_machines'] - 1:
        process_files(files_to_convert[int(len(files_to_convert) / config['n_machines']) * (config['machine_i']):])
        # we just save parameters once! In the last thread run by n_machine-1!
        config['normalize_mean'] = None
        config['normalize_std'] = None
        json.dump(config, open(config_file.DATA_FOLDER + config['audio_representation_folder'] + "config.json", "w"))
    else:
        first_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'])
        second_index = int(len(files_to_convert) / config['n_machines']) * (config['machine_i'] + 1)
        assigned_files = files_to_convert[first_index:second_index]
        process_files(assigned_files)

    print("Audio representation folder: " + config_file.DATA_FOLDER + config['audio_representation_folder'])
