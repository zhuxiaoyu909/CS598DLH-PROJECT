## GAT - Graph Attention Network (PyTorch) 
This repo is for the reproducibility study of CS598 Deep Learning for Healthcare from UIUC (:link: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)). <br/>

## Setup

To set up the environment, follow the steps below:

1. `git clone https://github.com/zhuxiaoyu909/CS598DLH-PROJECT.git`
2. Open Anaconda console and navigate into project directory `cd path_to_repo`
3. Run `conda env create` from project directory (this will create a brand new conda environment).
4. Run `activate pytorch-gat` (for running scripts from your console or setup the interpreter in your IDE)

It should execute environment.yml file which deals with dependencies. <br/>

-----

## Usage

### Training GAT
CORA:
To run it (from console) just call: <br/>
`python training_script_cora.py`

PPI:
To run it (from console) just call: <br/>
`python training_script_ppi.py`

* add the `--should_test` - to evaluate GAT on the test portion of the data
* add the `--enable_tensorboard` - to start saving metrics (accuracy, loss)

Performance evaluation results:
* On Cora I get the `81.6%` accuracy on test nodes
* On PPI I achieved the `0.973` micro-F1 score


The script will:
* Dump checkpoint *.pth models into `models/checkpoints/`
* Dump the final *.pth model into `models/binaries/`
* Save metrics into `runs/`, just run `tensorboard --logdir=runs` from your Anaconda to visualize it
* Periodically write some training metadata to the console

