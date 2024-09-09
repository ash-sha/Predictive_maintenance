import pandas as pd
import glob

def load_raw_data(dnames, rnames, input_path, output_path):
    for i, dname in enumerate(dnames):
        read_files = glob.glob(f"{input_path}/{dname}.txt")
        with open(f"{output_path}/{rnames[i]}.txt", "wb") as outfile:
            for f in read_files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read())

def load_processed_data(train_path, test_path, rul_path, col_names):
    df_train = pd.read_csv(train_path, sep='\s+', header=None, names=col_names)
    df_test = pd.read_csv(test_path, sep='\s+', header=None, names=col_names)
    df_test_RUL = pd.read_csv(rul_path, sep='\s+', header=None, names=['RUL'])
    return df_train, df_test, df_test_RUL
