# Sex differences in Myocardial Infarction prediction by Deep Learning methods

Currently, Myocardial Infarction (MI) is more often under-diagnosed for female patients than for male. Although, there is a gap in exploring the sex differences for Machine Learning models in this context.

This work aims at creating a framework for sex-related differences evaluation for CNN-based architectures for MI detection. We answer the question to what extent does sex of the patient influence the model performance.

The decision thresholds survey and separate training for male and female cohorts demonstrated noticeable differences in metrics including accuracy and sensitivity. However, bias testing was not able to highlight statistically significant differences, which also may be attributed to the testing data size.

We conclude that there are sex related differences in detecting MI for men and women for CNN-based models, which need to be carefully tested for every particular model, which may be done utilising our methods.

---

## Quick Start

Load data from: https://physionet.org/content/ptb-xl/1.0.3/

Mention the data path in the configuration file

```bash
git clone <repo>
cd <repo>
conda env create -f environment.yaml
conda activate <env_name>
```

## Running Experiments

To run the experiment choose config file from `configs/` and insert into `train.py` or `test.py` and run the file

---

## Experiments

Experiments may be found in the `experiments/` folder:

- `data_analysis.ipynb` - preparatory analysis of the PTB-XL dataset  
- `exp1_fcn.ipynb` and `exp1_resnet.ipynb` contain the results of trianning both models on the full train set  
- `exp2_fcn.ipynb` and `exp2_resnet.ipynb` contain the results of trianning both models on the female cohort  
- `exp3_fcn.ipynb` and `exp3_resnet.ipynb` contain the results of trianning both models on the male cohort  
- `exp4_CAM_FCN.ipynb` and `exp4_CAM_ResNet.ipynb` contain examples of CAM images for both healthy and MI cohorts (male and female)  
- `exp5_bootstrap_test_fcn.ipynb` `exp5_bootstrap_test_resnet.ipynb` contain results of the bootstrap bias test  
- `exp5_mannwhitneyu_test_fcn.ipynb` `exp5_mannwhitneyu_test_resnet.ipynb` contain results of the Mann-Whitney U bias test  

