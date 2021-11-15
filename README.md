# Polymer Named Entity Normalization

This repo contains code and data for the paper 'Machine-Guided Polymer Knowledge Extraction from the Literature Using Natural Language Processing: The Example of Named Entity Normalization' [[1]](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00554).

## Requirements and Setup

- Python 3.6+
- Pytorch (version 1.5.0)
- scikit-learn (version 0.22.1)
- spacy (version 2.1.6)

You can install all required Python packages using the provided env.yml file using `conda env create -f env.yml`

## Running the code

The code for normalization has been adapted from [https://github.com/iesl/expLinkage] [[2]](http://proceedings.mlr.press/v97/yadav19a.html). Some of the major changes include addition of the parameterized cosine distance metric and addition of a test mode for prediction of clusters for zero-shot data. The following commands can be used to replicate the experiments in the paper.

To train the supervised clustering model in our paper
```bash
python src/trainer/train_vect_data.py --config="src/utils/Config.py" --mode="train" --resultDir="/path/to/output_dir" --clusterFile="data/input_data/fastText/labeled_polymer_clusters.tsv"
```

To train the baseline model described in our paper
```bash
python src/baseline/baseline_train.py --labeled_file="ata/input_data/fastText/labeled_polymer_clusters_with_name.tsv" --use_labels=True --output_dir="path/to/output_dir"
```

## References

[1] Shetty, Pranav, and Rampi Ramprasad. "Machine-Guided Polymer Knowledge Extraction Using Natural Language Processing: The Example of Named Entity Normalization." Journal of Chemical Information and Modeling (2021).

[2] Yadav, Nishant, et al. "Supervised hierarchical clustering with exponential linkage." International Conference on Machine Learning. PMLR, 2019
