import logging
def log_metrics(mode, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('{} {}: {:.4f}'.format(mode, metric, metrics[metric]))

def normalize_question(question: str) -> str:
    question = question
    if question[-1] == '?':
        question = question[:-1]
    return question