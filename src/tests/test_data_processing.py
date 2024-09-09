import pytest
from src.data_processing.data_loader import load_raw_data
from os.path import exists

def test_load_data():
    dnames = ["train_FD001", "test_FD001", "RUL_FD001"]
    rnames = ["train", "test", "RUL"]
    input_path = "/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/raw/"
    output_path = '/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/interim/'

    load_raw_data(dnames,rnames,input_path,output_path)

    file_exists = exists("/Users/aswathshakthi/PycharmProjects/MLOps/Predictive_maintenance/data/interim/")
    assert file_exists, "file exists"