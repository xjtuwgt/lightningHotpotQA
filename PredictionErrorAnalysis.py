import gzip
import pickle
import json
import argparse
import os
import pandas as pd

from model_envs import MODEL_CLASSES
from plmodels.pldata_processing import Example, InputFeatures, get_cached_filename
from resultanalysis.errorAnalysis import error_analysis, data_analysis, error_analysis_question_type, prediction_score_analysis, prediction_score_gap_analysis
from envs import OUTPUT_FOLDER, DATASET_FOLDER

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--raw_data", type=str, default='data_raw/hotpot_dev_distractor_v1.json')
    parser.add_argument("--input_dir", type=str, default=DATASET_FOLDER, help='define output directory')
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    parser.add_argument("--pred_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')

    parser.add_argument("--graph_id", type=str, default="1", help='define output directory')
    parser.add_argument("--f_type", type=str, default='hgn_low', help='data type')

    # Other parameters
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default='train.graph.roberta.bs2.as1.lr2e-05.lrslayer_decay.lrd0.9.gnngat1.4.datahgn_docred_low_saeRecAdam.cosine.seed103', type=str,
                        help="Path to pre-trained model")

    parser.add_argument("--pred_res_name", default='dev_pred.json', type=str, help="Prediction result")

    parser.add_argument("--pred_score_name", default='dev_score.json', type=str, help="Prediction result")

    parser.add_argument("--error_res_name", default='error_res.json', type=str, help="error analysis result")

    parser.add_argument("--max_entity_num", default=60, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")

    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    print(config_class, model_class, tokenizer_class)
    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    # f_type = args.f_type
    #
    # cached_examples_file = os.path.join(args.input_dir,
    #                                     get_cached_filename('{}_examples'.format(f_type), args))
    # cached_features_file = os.path.join(args.input_dir,
    #                                     get_cached_filename('{}_features'.format(f_type),  args))
    # cached_graphs_file = os.path.join(args.input_dir,
    #                                  get_cached_filename('{}_graphs'.format(f_type), args))
    #
    # examples = pickle.load(gzip.open(cached_examples_file, 'rb'))
    # features = pickle.load(gzip.open(cached_features_file, 'rb'))
    # graph_dict = pickle.load(gzip.open(cached_graphs_file, 'rb'))
    #
    # example_dict = {example.qas_id: example for example in examples}
    # feature_dict = {feature.qas_id: feature for feature in features}

    raw_data_file = os.path.join(args.input_dir, args.raw_data)

    with open(raw_data_file, 'r', encoding='utf-8') as reader:
        raw_data = json.load(reader)

    # pred_results_file = os.path.join(args.pred_dir, args.model_type, 'pred.json')
    pred_results_file = os.path.join(args.pred_dir, args.model_name_or_path, args.pred_res_name)
    with open(pred_results_file, 'r', encoding='utf-8') as reader:
        pred_data = json.load(reader)

    pred_score_results_file = os.path.join(args.pred_dir, args.model_name_or_path, args.pred_score_name)
    with open(pred_score_results_file, 'r', encoding='utf-8') as reader:
        pred_score_data = json.load(reader)

    print('Loading predictions from: {}'.format(pred_results_file))
    print('Loading raw data from: {}'.format(args.raw_data))
    # print("Loading examples from: {}".format(cached_examples_file))
    # print("Loading features from: {}".format(cached_features_file))
    # print("Loading graphs from: {}".format(cached_graphs_file))

    # error_analysis(raw_data=raw_data, predictions=pred_data, tokenizer=None, use_ent_ans=False)
    # error_analysis_question_type(raw_data=raw_data, predictions=pred_data, tokenizer=None, use_ent_ans=False)
    # data_analysis(raw_data, example_dict, feature_dict, tokenizer, use_ent_ans=False)
    # metrics = hotpot_eval(pred_file, args.raw_data)
    # for key, val in metrics.items():
    #     print("{} = {}".format(key, val))
    df = prediction_score_analysis(raw_data=raw_data, predictions=pred_data, prediction_scores=pred_score_data)
    # df = prediction_score_gap_analysis(raw_data=raw_data, predictions=pred_data, prediction_scores=pred_score_data)

    error_res_results_file = os.path.join(args.pred_dir, args.model_name_or_path, args.error_res_name)
    df.to_json(error_res_results_file)
    print('saved {} records into {}'.format(df.shape, error_res_results_file))
    error_df = pd.read_json(error_res_results_file)
    print(error_df.shape)
