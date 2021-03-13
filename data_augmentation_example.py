import argparse
from os.path import join
from envs import DATASET_FOLDER
from dataugmentation.data_combined_processing import combine_data_graph_feat_examples, save_data_graph_feat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_path", type=str, default=join(DATASET_FOLDER, 'data_feat'))
    parser.add_argument("--model_type", default='roberta', type=str, help="Model type")
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()

    print('1. combine hgn with docred')
    combined_tag_type_pair_list_hgn_docred = [('train', 'hgn_low'), ('docred', 'docred_low')]
    hgn_docred_combined_type = 'hgn_docred_low'

    data_folder = args.data_path
    combine_data_graph_feat_examples(data_folder=data_folder, config=args, tag_f_type_list=combined_tag_type_pair_list_hgn_docred)



    # combined_tag_type_pair_list_hgn_sae_docred = [('train', 'hgn_low_sae'), ('docred', 'docred_low_sae')]
    # hgn_docred_sae_combined_type = 'hgn_docred_low_sae'
    #
    # combined_tag_type_pair_list_long_docred = [('train', 'long_low'), ('docred', 'docred_low')]
    # long_docred_combined_type = 'long_docred_low'
    # combined_tag_type_pair_list_long_sae_docred = [('train', 'long_low_sae'), ('docred', 'docred_low_sae')]
    # long_docred_sae_combined_type = 'long_docred_low_sae'


