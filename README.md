# Geometric Paratope Epitope Prediction
Official repository of the paper "Geometric Epitope and Paratope Prediction"

![](/figure/featured.png "")

## Environement

### IGEP
To create the IGEP environment, run the following command:

```bash
conda create -n IGEP python=3.8.* pip
conda activate IGEP
./env_IGEP.sh
```

### OGEP 
To create the OGEP environment, run the following command:

```bash
conda env create -f env_OGEP.yml
```

To run polymol, first you need to run this line on the terminal

```bash
apt-get update && apt-get install libgl1
```

## Preprocessing
To replicate the results of the paper, we downloaded all the necessary pdb form the Sabdab database (http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?all=true). See also Epipred (https://pubmed.ncbi.nlm.nih.gov/24753488/) for more details.

### Data
You can download the data from [here](https://drive.google.com/file/d/1BugJmGU6-AI3wDNh9YdCBHA4WprrLG7z/view?usp=sharing). 

### Residues
To process the pdb files and obtain the input features and graph representation, run the following command:

```bash
python Processing/process_features.py --path data/with/pdb/files --csv_name csv_file.csv
```
where --path specify the path with the pdb files and --csv_name the name of the csv file with the pdb to process.
The script will generate a processed-dataset.p pickle file with the processed data.

### Surface
To compute the surface and then transfer the features on it run:

```bash
python Processing/pdb2surface.py --pdb-folder data/with/pdb/files --dest-folder data/with/wrl/files
``` 
where --pdb-folder specify the path with the pdb files and the processed data pickle file, while --dest-folder the path where to save the wrl files.

## IGEP
To run the IGEP model, run the following command:

``` bash
conda activate IGEP
python train_IGEP.py --json-file params.json --json-keys common EPMP
```
where --json-file specify the path with the json file with the parameters and --json-keys the keys of the parameters to use.

You can test the results using the notebook test_IGEP.ipynb.

## OGEP
To run the OGEP model, run the following command:

``` bash
conda activate OGEP
python train_OGEP.py -p params.json --model diffNet
```
where -p specify the path with the json file with the parameters and --model the model to use (PiNet, diffNet).

You can test the results using the notebook test_OGEP.ipynb.

## Cite
If you use this code, please cite the following paper:

```bibtex
@article{pegoraro2023geometric,
  title={Geometric Epitope and Paratope Prediction},
  author={Pegoraro, Marco and Domin{\'e}, Cl{\'e}mentine and Rodol{\`a}, Emanuele and Veli{\v{c}}kovi{\'c}, Petar and Deac, Andreea},
  journal={bioRxiv},
  year={2023},
}
```

