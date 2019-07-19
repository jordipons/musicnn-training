# musicnn-training

In this repository you will find the Tensorflow code for training deep convolutional neural networks for music audio tagging. We employed this code for traininc [musicnn](https://github.com/jordipons/musicnn/), a musically motivated convolutional neural network for music audio tagging.

Reference:
```
@inproceedings{pons2018atscale,
  title={End-to-end learning for music audio tagging at scale},
  author={Pons, Jordi and Nieto, Oriol and Prockup, Matthew and Schmidt, Erik M. and Ehmann, Andreas F. and Serra, Xavier},
  booktitle={19th International Society for Music Information Retrieval Conference (ISMIR2018)},
  year={2018},
}

```

## How to?

#### Installation:
Create a python 3 virtual environment and install dependencies `pip install -r requirements.txt`

Install tensorflow for CPU `pip install tensorflow` or for CUDA-enabled GPU `pip install tensorflow-gpu`

#### Download data:
Download [US8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), and ASC-TUT dataset ([dev-set](https://zenodo.org/record/400515#.W9n2UtGdZhE) / [eval-set](https://zenodo.org/record/1040168#.W9n2jNGdZhE)).

#### Preprocess the data:
To preprocess the data, first set some `config_file.py` variables:
- `DATA_FOLDER`, where you want to store all your intermediate files (see folders structure below).
- `config_preprocess['audio_folder']`, where your dataset is located.

Preprocess the data running `python preprocess.py asc_spec`. Note `asc_spec` config option is defined in `config_file.py`

After running `preprocess.py`, spectrograms are in `../DATA_FOLDER/audio_representation/asc__time-freq/`

_*Warning!*_ Rename `index_0.tsv` to `index.tsv`. This is because this script is parallelizable.

#### Regularized deep learning results:

Set `config_sl` dictionary in `config_file.py`, and run `CUDA_VISIBLE_DEVICES=0 python sl_train.py spec`

Once training is done, the resulting model is stored in `../DATA_FOLDER/experiments/fold_0_1541174334/`

To evaluate the model, run `CUDA_VISIBLE_DEVICES=0 python sl_evaluate.py fold_0_1541174334`

## Scripts

**Configuration** and preprocessing scripts:
- `config_file.py`: file with all configurable parameters.
- `preprocess.py`: pre-computes and stores the spectrograms.

Scripts for **regularized deep learning models** experiments:
- `sl_train.py`: run it to train your model. First set `config_sl` in `config_file.py`
- `sl_evaluate.py`: run it to evaluate the previously trained model.
- `models_sl.py`: script where the architectures are defined.

## Folders structure

- `/src`: folder containing previous scripts.
- `/aux`: folder containing auxiliar additional scripts. These are used to generate the index files in `/data/index/`.
- `/data`: where all intermediate files (spectrograms, results, etc.) will be stored. 
- `/data/index/`: indexed files containing the correspondences between audio files and their ground truth.

When running previous scripts, the following folders will be created:
- `./data/audio_representation/`: where spectrogram patches are stored.
- `./data/experiments/`: where the results of the experiments are stored.
