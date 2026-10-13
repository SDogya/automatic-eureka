"""TraFL experiment package: modules for data, sampling, objective and execution.

Public functions remain in their logical modules. Importing the package alone
performs no model loading, CUDA initialization, data download or training.
Use python run.py --config config.yaml --mode train|eval|smoke as the entry point.
"""
