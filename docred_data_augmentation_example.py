import argparse
from os.path import join
from envs import DATASET_FOLDER
from dataugmentation.data_combined_processing import combine_data_graph_feat_examples, save_data_graph_feat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_path", type=str, default=join(DATASET_FOLDER, 'data_feat'))
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()
    data_folder = args.data_path
    for key, value in vars(args).items():
        print('Parameter {}: {}'.format(key, value))
    print('1. combine hgn with docred')
    combined_tag_type_pair_list_hgn_docred = [('train', 'hgn_low'), ('docred', 'docred_low')]
    hgn_docred_combined_type = 'hgn_docred_low'
    features, examples, graphs = combine_data_graph_feat_examples(data_folder=data_folder, config=args, tag_f_type_list=combined_tag_type_pair_list_hgn_docred)
    save_data_graph_feat(out_folder=join(data_folder, 'train'), config=args, f_type=hgn_docred_combined_type,
                         examples=examples, features=features, graphs=graphs)

    print('2. combine hgn with docred sae')
    combined_tag_type_pair_list_hgn_sae_docred = [('train', 'hgn_low_sae'), ('docred', 'docred_low_sae')]
    hgn_docred_sae_combined_type = 'hgn_docred_low_sae'
    features, examples, graphs = combine_data_graph_feat_examples(data_folder=data_folder, config=args,
                                                                  tag_f_type_list=combined_tag_type_pair_list_hgn_sae_docred)
    save_data_graph_feat(out_folder=join(data_folder, 'train'), config=args, f_type=hgn_docred_sae_combined_type,
                         examples=examples, features=features, graphs=graphs)

    print('3. combine long with docred')
    combined_tag_type_pair_list_long_docred = [('train', 'long_low'), ('docred', 'docred_low')]
    long_docred_combined_type = 'long_docred_low'
    features, examples, graphs = combine_data_graph_feat_examples(data_folder=data_folder, config=args,
                                                                  tag_f_type_list=combined_tag_type_pair_list_long_docred)
    save_data_graph_feat(out_folder=join(data_folder, 'train'), config=args, f_type=long_docred_combined_type,
                         examples=examples, features=features, graphs=graphs)

    print('4. combine long with docred sae')
    combined_tag_type_pair_list_long_sae_docred = [('train', 'long_low_sae'), ('docred', 'docred_low_sae')]
    long_docred_sae_combined_type = 'long_docred_low_sae'
    features, examples, graphs = combine_data_graph_feat_examples(data_folder=data_folder, config=args,
                                                                  tag_f_type_list=combined_tag_type_pair_list_long_sae_docred)
    save_data_graph_feat(out_folder=join(data_folder, 'train'), config=args, f_type=long_docred_sae_combined_type,
                         examples=examples, features=features, graphs=graphs)


    ### python3 docred_data_augmentation_example.py --model_type 'roberta'
    ### python3 docred_data_augmentation_example.py --model_type 'albert'