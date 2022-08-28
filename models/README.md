A note on naming: 

Models are named according to the number of samples in the training set, so for example:

lifespan_12K_57sites_train means the training sample consisting of 12000 subjectes from 57 sites. Modality = Cortical thickness (Destrieux atlas, aparc.a2009s) and subcortical volumes (aseg).

mqc: manual qc

mqc2: a subset of the mqc sample matched to the training data from the big dataset (i.e. which allows a fair comparison between the models).

lifespan_yeo17_15K_45sites.tar.gz: Yeo-17 network connectomes for resting-state fMRI. Training set (80% of sample) for evaluation on test set (20% of sample).

lifespan_yeo17_22K_45sitest.tar.gz: Yeo-17 network connectomes for resting-state fMRI. Trained on full sample, use to transfer to new datasets, not included in inital training.

surfacearea_20K_66sites_train.zip: Surface area models (Destrieux atlas, aparc.a2009s), using a similar data set as [eLife](https://elifesciences.org/articles/72904) paper. Training data set (50% of sample, 20K subjects), testing (50%, 20K subjects) data set used to evaluate. 

surfacearea_40K_66sites.zip: Surface area models, full training set (40K subjects), use to transfer to new datasets (sites) that were not included in the inital training.