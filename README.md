# Uncertainty Estimation for Change Point Detection models

### Final Projects of the Machine Learning Course, 2023
Alexander Stepikin, Maria Kovaleva, Alexander Kukharskii

<ins>Abstract</ins>:

Change Point Detection (CPD) aims to identify moments of abrupt data distribution shifts in sequential data. 
While there exist deep learning CPD models, they typically are not able to provide any confidence in their predictions which is often required by real-world applications. 
In this project, we study ensembles of deep change point detectors and develop two ways of taking into account uncertainty in the predicted CP probabilities to produce robust estimates of the change points. 
Experiments conducted on synthetic and real-world datasets suggest that small ensembles of deep detectors outperform single-model CPD baselines. 
Uncertainty-aware aggregation of the change point scores obtained by an ensemble with CUSUM-statistic is proved to be beneficial in case of semi-structured high-dimensional data, such as video clips with explosions. 
The proposed rejection-based procedure helps to decrease the amount of false alarms, thus, optimizing one of the principled CPD metrics.

This project is based on the [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548182) by E.Romanenkova, et.al. "Indid: Instant disorder detection via a principled neural network" (ACM Multimedia '22). Thus, we use the general CPD pipeline developed in this work while complementing it with additional experiments. In particular, we add ```ensembles.py``` module to work with ensembles of Change Point detectors and update ```metrics.py``` module to be consistent with the new proposed models.

One can find the list or the necessary packages in the ```requirements.txt``` file. The datasets are available [online](https://disk.yandex.ru/d/_PQyni3AhyLu5g). Once loaded, they should be placed into the ```data/``` folder which is currently empty. We also provide the weights for the base models we trained in ensembles. They are available by the [link](https://disk.yandex.ru/d/Dn9pPGMDKBL-kg). Once loaded, they should be placed into the ```saved_models/``` folder.

In order to reproduce our experiments, please look at the ```CPD_Ensembles.ipynb``` notebook which runs all the main experiments on the dataset of sequences of MNIST images. Note that they can be run for any other dataset or base model available in this pipeline. For instructions, please, go to this notebook.

The file ```UE for CPD.pdf``` contains the presentation of our project.
