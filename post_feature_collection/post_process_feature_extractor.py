import argparse
from model_envs import MODEL_CLASSES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--score_file_path", type=str, required=True)
    # Other parameters