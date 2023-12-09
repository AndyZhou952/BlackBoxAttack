# Spear vs Shield: Offensive and Defensive Strategies in Black-Box Adversarial Attacks

This proejct was created for Duke University ECE 685D 2023 Fall. 

Our study utilizes the three attck algorithms (query-limited, partial-info, label-only) proposed by Ilya et al. (2018) and the defense AAA algorithm by Chen et al. (2022) to explore both offensive and defensive aspects of Black-Box Adversarial Attacks. We develop these algorithms implemented via the PyTorch framework.

## File Description:

1. `algo`:

(1)`__init__.py`: Necessary packages

(2) `attacker.py`: Contains functions for NES gradient estiamtes and generating adversarial images (for all attack setting)

(3) `defender.py`: Contains the AAA protected classifier definition and the partial info masking.

2. `model`:

(1) `states`: This folder contains the trained models `butterfly_classifier.pth` and `imagenetclassifier` for both the Butterfly and the ImageNet datasets.

(2) `butterfly_classifier.py`: Butterfly classifier definition, utilizing the pretrained `densenet121` with three additional fully connected layers.

(3) `imagenet_classifier.py`: ImageNet classifier definition, utilizing the pretrained `inception_v3`  with three additional fully connected layers.

3. `utils`:

(1) `__init__.py`: Necessary packages

(2) `base.py`: Methods to train classifiers, evaluate accuracy, and make predictions.

(3) `data.py`: `ImageDataset` definition, create the Butterfly and ImageNet datasets with mapping and sample image dictionary for each class.


## Reproducibility:

1. First, download the datasets from Kaggle and run `train_models.ipynb`.

2. Run `eval_attacker-Butterfly.ipynb`, `eval_attacker-ImageNet.ipynb` to gather results from the three attack mechanisms.

3. Run `eval_attacker_defender-Butterfly.ipynb`, `eval_attacker_defender-ImageNet.ipynb` to gather results.

4. Finally, run `results.ipynb` to get the results and the visualizations.


