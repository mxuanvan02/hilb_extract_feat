import glob
import pickle

import numpy as np
import pandas as pd
from zipfile import ZipFile

def get_PSSM(zip_file_name, file_name):
    with ZipFile(zip_file_name) as z:
        values = []
        with z.open(file_name) as f:
            readf = f.readlines()
            lines = [line.decode("utf-8") for line in readf]
            for i in range(3, 23):
                values.append(lines[i].strip().split('\t')[1:])
        return np.array(values).T

def make_dataset(path_to_datset_dir):
    def make_from_pairs(df_pairs: pd.DataFrame, prefix_save):
        list_pssm_A = []
        for prot_id in df_pairs['proteinA']:
            list_pssm_A.append(get_PSSM(path_to_datset_dir + "/PSSM_profile.zip", prot_id + ".pssm"))
        pickle.dump(list_pssm_A, open(prefix_save + "_A_pssm.pkl", "wb"))
        print("Number of PSSMs;", len(list_pssm_A))
        print("Size of the first PSSM;", list_pssm_A[0].shape)
        print("Saved")

        list_pssm_B = []
        for prot_id in df_pairs['proteinB']:
            list_pssm_B.append(get_PSSM(path_to_datset_dir + "/PSSM_profile.zip", prot_id + ".pssm"))
        pickle.dump(list_pssm_B, open(prefix_save + "_B_pssm.pkl", "wb"))
        print("Number of PSSMs;", len(list_pssm_B))
        print("Size of the first PSSM;", list_pssm_B[0].shape)
        print("Saved")

        return 0

    print("\nGet PSSM ...")

    pos_pairs = pd.read_csv(path_to_datset_dir + '/positive.txt', sep=',')
    make_from_pairs(pos_pairs, 'pos')
    neg_pairs = pd.read_csv(path_to_datset_dir + '/negative.txt', sep=',')
    make_from_pairs(neg_pairs, 'neg')

    return 0


if __name__ == "__main__":
    make_dataset("dataset/Yeasts")
