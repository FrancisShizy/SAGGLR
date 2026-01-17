# data_preprocess/data_parser_utils.py
import argparse

def data_parser():
    parser = argparse.ArgumentParser(description="Data preprocessing & scaffold analysis")

    parser.add_argument("--seed", type=int, default=1337, help="seed")
    parser.add_argument("--data_path", nargs="?", default="data_preprocess/data", help="Output data path.")
    parser.add_argument("--test_set_fraction", type=float, default=0.2, help="test set size (ratio)")
    parser.add_argument("--val_set_fraction", type=float, default=0.1, help="val split fraction from training")
    parser.add_argument("--scaffold_analysis_root", type=str, default="data_preprocess/scaffold_analysis",
                        help="where to save one-time split scaffold analysis")


    return parser