# Sub-models for risk and resilience assessments at a regional scale

**Rincon, R., Padgett, J.E.**

Different sub-models are needed to perform regional-scale risk and resilience assessments of infrastructure systems. We share some models commonly used to perform such analyses in this repository. Currently, this is intended to be shared as a set of scripts that can be accessed and modified once you have copied them to your local machine (rather than a python package). Importing the modules or functions follows the typical structure in Python.

You can install the requirements directly in your preferred environment or use the environment shared in the repository. For the latter, clone this repository to your local machine. Create a conda environment using the **risk_resilience.yml** file. For this, follow these steps: 
- Using a command line interface, go to your repository folder ('\risk_resilience_models\configs').
- Then, create the environment `risk_resilience` (the name is automatically given by the .*yml file)
- Activate the environment in your IDE. 

Previous steps are done with the following commands:

```shell
cd "path/of/your/repository"
conda env create -f risk_resilience.yml
conda activate risk_resilience
```

## Module for probabilistic and scenario-based hazard analysis

This is a module for seismic hazard analyses in academic examples. It enables the creation of hypothetical point and line faults, sampling earthquake events, and generating correlated ground motion intensity measures. It has implemented only one ground motion model (AB95) and one intensity measure correlation model (JB08).

To see an example on how to use it, check the following jupyter notebooks:
- [Hazard analyses examples](notebooks/hazard_example.ipynb)


## Module for fragility functions derivations using responses of systems with multiple failures modes

_Under development_



### If you are collaborating

If you install any new package, create a .*yml file and replace it with the one in the repository for future reproducibility.

```shell
conda activate risk_resilience
conda env export > risk_resilience.yml
```

### Was the conda environment updated?

If you have a new file for the environment and want to update the previous 'risk_resilience.yml', use the new_file.yml and run the following command:

```shell
conda activate risk_resilience
conda env update --file new_file.yml --prune
```
