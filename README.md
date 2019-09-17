# musicnn-training

In this repository you will find the Tensorflow code for training deep convolutional neural networks for audio-tagging.

We employed this code for training [musicnn](https://github.com/jordipons/musicnn/), a set of pre-trained deep convolutional neural networks for music audio tagging.


## Installation:
`git clone https://github.com/jordipons/musicnn-training.git`

Create a python3 virtual environment `python3 -m venv env`, activate it `source ./env/bin/activate`, and install the dependencies `pip install -r requirements.txt`

Install tensorflow for CPU `pip install tensorflow==1.14.0` or for CUDA-enabled GPU `pip install tensorflow-gpu==1.14.0`. Note this code was developed following Tensorflow 1 API, not Tensorflow 2.

## Usage

#### Download a music audio tagging dataset:
For example, download the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset.

#### Preprocess the dataset:
To preprocess the data, first set some `config_file.py` variables:
- `DATA_FOLDER`, where you want to store all your intermediate files (see folders structure below).
- `config_preprocess['audio_folder']`, where your dataset is located.

Preprocess the data acessing to `src/` and running `python preprocess_librosa.py mtt_spec`. The `mtt_spec` option is defined in `config_file.py`

After running `preprocess_librosa.py`, the computed spectrograms are in `../DATA_FOLDER/audio_representation/mtt__time-freq/`

_*Warning!*_  
Rename `../DATA_FOLDER/audio_representation/mtt__time-freq/index_0.tsv` to `index.tsv`. This is because this script is parallelizable. In case you parallelized the pre-processing accross several machines, copy the files in the same direcotry and run `cat index* > index.tsv`

#### Train and evaluate a model:

Define the training parameters by setting the `config_train` dictionary in `config_file.py`, and run `CUDA_VISIBLE_DEVICES=0 python train.py spec`. The `spec` option is defined in `config_file.py`

Once training is done, the trained model is stored in, e.g.: `../DATA_FOLDER/experiments/1563524626spec/`

To evaluate the model, run `CUDA_VISIBLE_DEVICES=0 python evaluate.py 1563524626spec`

## Scripts

**Configuration and preprocessing** scripts:
- `config_file.py`: file with all configurable parameters.
- `preprocess_librosa.py`: pre-computes and stores the spectrograms.

Scripts for **running deep learning experiments**:
- `train.py`: run it to train your model. First set `config_train` in `config_file.py`
- `evaluate.py`: run it to evaluate the previously trained model.
- `models.py`, `models_baselines.py`, `models_frontend.py`, `models_midend.py`, `models_backend.py`: scripts where the architectures are defined.

**Auxiliar** scripts:
- `shared.py`: script containing util functions (e.g., for plotting or loading files).
- `train_exec.py`: script to successively run several pre-configured experiments.

## Folders structure

- `/src`: folder containing previous scripts.
- `/aux`: folder containing additional auxiliar scripts. These scripts are used to generate the index files for each dataset. The index files are already computed in `/data/index/`
- `/data`: where all intermediate files (spectrograms, results, etc.) will be stored. 
- `/data/index/`: indexed files containing the correspondences between audio files and their ground truth. Index files for the  [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset (`mtt`) and  the [Million Song Dataset](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split) (`msd`) are already provided. 

When running previous scripts, the following folders will be created:
- `./data/audio_representation/`: where spectrogram patches are stored.
- `./data/experiments/`: where the results of the experiments are stored.

## References:
```
@inproceedings{pons2018atscale,
  title={End-to-end learning for music audio tagging at scale},
  author={Pons, Jordi and Nieto, Oriol and Prockup, Matthew and Schmidt, Erik M. and Ehmann, Andreas F. and Serra, Xavier},
  booktitle={19th International Society for Music Information Retrieval Conference (ISMIR2018)},
  year={2018},
}
```
```
@inproceedings{pons2019musicnn,
  title={musicnn: pre-trained convolutional neural networks for music audio tagging},
  author={Pons, Jordi and Serra, Xavier},
  booktitle={Late-breaking/demo session in 20th International Society for Music Information Retrieval Conference (LBD-ISMIR2019)},
  year={2019},
}

```
