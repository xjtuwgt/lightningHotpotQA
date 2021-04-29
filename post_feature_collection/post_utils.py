from leaderboardscripts.lb_postprocess_utils import row_x_feat_extraction

def get_best_f1_intervals(scores, labels):
    best_f1_intervals = []
    for s, l in zip(scores, labels):
        l1 = [1 if i in l else 0 for i in range(len(s))]
        sorted_sl = sorted(zip(s, l1), key=lambda x: x[0])
        max_f1 = 0
        max_f1_scores = []
        for i in range(len(sorted_sl)):
            tp_left = sum([x[1] for x in sorted_sl[i:]])
            prec_left = tp_left / (len(sorted_sl) - i)
            recall_left = tp_left / (len(l) + 1e-6)
            f1_left = 2 * prec_left * recall_left / (prec_left + recall_left + 1e-6)
            if f1_left > max_f1 + 1e-6:
                max_f1 = f1_left
                max_f1_scores = [sorted_sl[i][0] - 1e-6]
            elif abs(f1_left - max_f1) < 1e-6:
                max_f1_scores.append(sorted_sl[i][0] - 1e-6)
            tp_right = sum([x[1] for x in sorted_sl[i+1:]])
            prec_right = tp_right / (len(sorted_sl) - i - 1 + 1e-6)
            recall_right = tp_right / (len(l) + 1e-6)
            f1_right = 2 * prec_right * recall_right / (prec_right + recall_right + 1e-6)
            if f1_right > max_f1 + 1e-6:
                max_f1 = f1_right
                max_f1_scores = [sorted_sl[i][0] + 1e-6]
            elif abs(f1_right - max_f1) < 1e-6:
                max_f1_scores.append(sorted_sl[i][0] + 1e-6)
        best_f1_intervals.append((max_f1, (min(max_f1_scores), max(max_f1_scores))))
    return best_f1_intervals