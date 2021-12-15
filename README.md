# braincharts

[![Gitter](https://badges.gitter.im/predictive-clinical-neuroscience/community.svg)](https://gitter.im/predictive-clinical-neuroscience/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Documentation Status](https://readthedocs.org/projects/pcntoolkit/badge/?version=latest)](https://pcntoolkit.readthedocs.io/en/latest/?badge=latest)

### Pre-trained models, code, supporting files for:
### [Charting Brain Growth and Aging at High Spatial Precision](https://www.biorxiv.org/content/10.1101/2021.08.08.455487v2)

**Training the reference cohort** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/predictive-clinical-neuroscience/braincharts/blob/master/scripts/fit_normative_models.ipynb)

**Fit pre-trained model to new data** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/predictive-clinical-neuroscience/braincharts/blob/master/scripts/apply_normative_models.ipynb)

**Abstract:** Defining reference models for population variation, and the ability to study individual deviations is essential for understanding inter-individual variability and its relation to the onset and progression of medical conditions. In this work, we assembled a reference cohort of neuroimaging data from 82 sites (N=58,836; ages 2-100) and use normative modeling to characterize lifespan trajectories of cortical thickness and subcortical volume. Models are validated against a manually quality checked subset (N=24,354) and we provide an interface for transferring to new data sources. We showcase the clinical value by applying the models to a transdiagnostic psychiatric sample (N=1,985), showing they can be used to quantify variability underlying multiple disorders whilst also refining case-control inferences.  These models will be augmented with additional samples and imaging modalities as they become available. This provides a common reference platform to bind results from different studies and ultimately paves the way for personalized clinical decision making. 

### **Interactive visualizations of evaluation metrics:**

**1. Full test set (including 10 randomized split halfs)**

Explained Variance [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/10foldCV_EVviz.ipynb)

MSLL [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/10foldCV_MSLLviz.ipynb)

Kurtosis [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/10foldCV_Kurtosisviz.ipynb)

Skew [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/10foldCV_Skewviz.ipynb)

**2. mQC test set**

Explained Variance [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/mQC_EVviz.ipynb)

MSLL [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/mQC_MSLLviz.ipynb)

Kurtosis [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/mQC_Kurtosisviz.ipynb)

Skew [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/mQC_Skewviz.ipynb)

**3. Patients test set**

Explained Variance [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/patients_EVviz.ipynb)

MSLL [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/patients_MSLLviz.ipynb)

Kurtosis [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/patients_Kurtosisviz.ipynb)

Skew [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/patients_Skewviz.ipynb)

**4. Transfer test set**

Explained Variance [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/transfer_EVviz.ipynb)

MSLL [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/transfer_MSLLviz.ipynb)

Kurtosis [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/transfer_Kurtosisviz.ipynb)

Skew [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saigerutherford/brainviz-app/blob/main/transfer_Skewviz.ipynb)
