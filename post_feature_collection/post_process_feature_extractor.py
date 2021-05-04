from post_feature_collection.post_process_argument_parser import train_parser

if __name__ == '__main__':
    args = train_parser()

    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))