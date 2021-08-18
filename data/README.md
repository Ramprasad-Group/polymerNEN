## Description of datasets

The folder input_data contains the input files needed for training a supervised clustering model and the baseline model. Input files are provided for Word2Vec and fastText. Each of the files are described below.

- `labeled_polymer_clusters.tsv` contains data in the format `<point_ID> <cluster_ID> <feature_1>...<feature_n>` and is the input file for supervised clustering.

- `labeled_polymer_clusters_with_name.tsv` contains data in the format `<point_ID> <cluster_ID> <entity_name> <feature_1>...<feature_n>` and is the input file for training the baseline model.

The generated_datasets folder contains the datasets generated as part of this work.

- `PNE_list` is a list of polymer named entities. This can be used as a weak supervision source for training NER models. Semicolon is used as a separator in this file.

- `predicted_labels.csv` is a list of clusters predicted by applying the supervised clustering model to the unlabeled set.

