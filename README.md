NO-Toxicity
==============================

##  Speech Toxicity Classifier using LSTM
In this project, the main goal was to classify a part of speech as toxic or not and if toxic then what type of toxicity. The model is trained using LSTM layers connected with Fully Connected layers and is implemented using pytorch.


## Setup
Make sure you have the required python dependencies for running the project.
1. Run `pip3 install -r requirements.txt` in your working environment to install the required pacakages.
2. Download the glove embeddings from https://www.kaggle.com/danielwillgeorge/glove6b100dtxt and place it in the location `NO_Toxicity.src.models`.

I have not included pytorch package in the requirements.txt, so please install the pytorch version according to your cuda and cudnn versions. It is recommeneded to run the project on GPU(if available) for lower run time. Otherwise, project will default to CPU.

## How to produce the same result?
- First, change the name of root directory to `no_toxicity`.
- Open the notebook name `1.no-toxicity.ipynb` present in notebooks folder.
- Now run the cells and follow the comments in it.
- You can see the respective source modules to see their implementation.

## Future Works
Working on the deployment of the project to take user's input and ouput the respective translation.


## Contributing
As this project is still under work, it would be much appericated if you submit a PR for cleanups, error-fixing, or adding new (relevant) content.


--------


