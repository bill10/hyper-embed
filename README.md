# hyper-embed

## Usage
To use this package, copy the `hyperEmbed.py` file to your working directory, and follow the example `medline.ipynb` notebook.

Currently, the input data has to be prepared in a specific form as follows.
* The data is organized as a few subdirectories. Each sub directory will be read in as one hypergraph and the subdirectory name is used as the timestamp for the snapshot. The timestamps (subdirectory names) have to be integers but don't have to be consecutive.
* Each subdirectory contains one or more data files. Each line in the file corresponds to a hyperedge. Fields in each line can be separated by any character. The first field is edge ID, and remaining fields are node IDs in the hyperedge. Different lines can have different number of fields. Node IDs have to be integers between 0 and number_of_nodes-1. Edge IDs can be anything.
* An example folder structure is as follows
```
MEDLINE
| 1990
  | part0.csv
  | part1.csv
| 1991
  | part.csv
```
