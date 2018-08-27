# Event Embedding Models #

This is a Keras implementation of different event embedding models. 


## Quick summary
* This repository contains role based word embedding, event embedding models and evaluations using neural network. All models are constructed on role based word embedding which make use of tensor factorization technique designed by Ottokar Tilk. 
* Currently following models are implemented:
    * Non-incremental Role Filler Model (NNRF)
    * Multi-task Role Filler Model (NNRF-MT)
    * Role Averaging Model (RoFA)
    * Residual Role Averaging Model (ResRoFA)
* More details are in comments of each file.


## How do I get set up? 
### Deployment
* The root directory should be modified in `config.py`.
* $PYTHONPATH of this repository is configured.

#### Dependencies
* Python 2.7
* CUDA 7.5
* Tensorflow 0.10
* Keras 2.0

#### Instructions
* Using docker is a better way to deploy the models. 
* [Docker hub repository](https://hub.docker.com/r/tonyhong/event-embedding/) for quick deployment. 
* After running the docker image, please follow the instructions in [docker hub repository](https://hub.docker.com/r/tonyhong/event-embedding/) to install `h5py`, `nltk` and download `wordnet` for `nltk`. 

Example steps to install docker and the images:
1. Follow the docker-ce installation instructions on the docker web site, e.g., for [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/), if you don't already have docker installed.
1. Once you have installed docker, assuming you also set up the docker group for your own username, download the docker image with:
   *  `$ docker pull tonyhong/event-embedding:paper`
1. Enter the docker image with:
   *  `$ docker run -ti tonyhong/event-embedding:paper`
   
   you should now be at a root prompt inside a running docker container.
1. Install `h5py` and `nltk` with the following command:
   *  `$ pip install h5py nltk`
1. Install the WordNet repository with:
   *  `$ ipython`
   *  `$ nltk.download()`
   
   and go through the menus to download the wordnet data. 
1. Install ssh and git with:
   *  `$ apt update`
   *  `$ apt install git ssh`
   

#### File arrangements
* All files should be organized in following structure:
```
.
├── data
├── model
├── eval_data
└── event-embedding
    ├── main.py
    ├── config.py
    ├── utils.py
    ├── batcher.py
    ├── model
    │   ├── __init__.py
    │   ├── embeddings.py
    │   ├── layers.py
    │   ├── generic.py
    │   ├── nnrf.py
    │   ├── nnrf_mt.py
    │   ├── rofa.py
    │   ├── resrofa.py
    │   ├── rofa_st.py
    │   └── resrofa_st.py
    ├── evaluation
    │   ├── __init__.py
    │   └── ...
    └── README.md
```

* Pre-trained models are avaiable [here](https://drive.google.com/open?id=1B05aCqf96QvlophDpCCDvvNPw2MOgIGI). These files should be placed in ```./model/test/``` .

* Evaluation data files are avaiable [here](https://drive.google.com/open?id=1B05aCqf96QvlophDpCCDvvNPw2MOgIGI). These files should be placed in ```./eval_data/``` .

* The training data should be placed in ```./data/```. If you need the training data, please contact the admin directly. 


### How to run tests?
* Train a model using Tensorflow:
    * `$ python main.py NAME_OF_MODEL VERSION`
* Train a model using Theano:
    * `$ KERAS_BACKEND=theano python main.py NAME_OF_MODEL VERSION`
* Eval a model:
   * `$ python evaluation/EVAL_SCRIPT.py NAME_OF_MODEL VERSION`
   * e.g. `$ python evaluation/eval_pado_mcrae.py NNRF oct_e75`


## Admin
* [Tony Hong](https://github.com/tony-hong)


## Reference 
* [RoleFiller](https://git.sfb1102.uni-saarland.de/asayeed/RoleFiller/)
* [SingleOutputModel](https://git.sfb1102.uni-saarland.de/asayeed/SingleOuputModel)
* [TensorFlow](https://github.com/tensorflow/tensorflow)
* [Keras](https://github.com/fchollet/keras)
* [Pado 2007](http://scidok.sulb.uni-saarland.de/volltexte/2007/1138/) Padó, U., 2007. The integration of syntax and semantic plausibility in a wide-coverage model of human sentence processing.
* [Baroni 2010](http://www.mitpressjournals.org/doi/abs/10.1162/coli_a_00016#.WPh7aFOGPVo) Baroni, M. and Lenci, A., 2010. Distributional memory: A general framework for corpus-based semantics. Computational Linguistics, 36(4), pp.673-721.
* [Sayeed 2015](http://ai2-s2-pdfs.s3.amazonaws.com/3fdd/125837c75a3963641f8db801d8f014089830.pdf) Sayeed, A., Demberg, V. and Shkadzko, P., 2015. An exploration of semantic features in an unsupervised thematic fit evaluation framework. Italian Journal of Computational Linguistics, 1(1).
* [Greenberg 2015](https://www.researchgate.net/profile/Vera_Demberg/publication/301404462_Improving_unsupervised_vector-space_thematic_fit_evaluation_via_role-filler_prototype_clustering/links/5756a0ae08ae10c72b697f11.pdf) Greenberg, C., Sayeed, A.B. and Demberg, V., 2015. Improving unsupervised vector-space thematic fit evaluation via role-filler prototype clustering. In HLT-NAACL (pp. 21-31).
* [Tilk 2016](https://pdfs.semanticscholar.org/d08d/663d7795c76bb008f539b1ac7caf8a9ef26c.pdf) Tilk, O., Demberg, V., Sayeed, A., Klakow, D. and Thater, S., 2016
