import numpy as np

def compute_intersection(ground_truth, preds, threshold):
    # Identifica os ids unicos para fazer a matriz IDF1
    parts_true = np.array([list(map(float, x.split(','))) for x in ground_truth])
    # f_true = parts_true[:, 0] - 1  
    # x_true = parts_true[:, 2]
    # y_true = parts_true[:, 3]
    # w_true = parts_true[:, 4]
    # h_true = parts_true[:, 5]
    unique_true_ids = np.unique(parts_true[:,1])

    parts_pred = np.array([list(map(float, x.split(','))) for x in preds])
    # f_preds = parts_pred[:, 0] - 1  
    # x_preds = parts_pred[:, 2]
    # y_preds = parts_pred[:, 3]
    # w_preds = parts_pred[:, 4]
    # h_preds = parts_pred[:, 5]
    unique_preds_ids = np.unique(parts_pred[:,1])
    IDF1 = np.zeros((unique_preds_ids.shape[0], unique_true_ids.shape[0]))

    # Percorre todos os frames
    for i in range(max(parts_true[:,0])):
        pass        


def greedy_match(iou, predicts, trues):
    iou = iou.copy()
    # Faz o casamento de forma gulosa
    matches_iou = np.full(shape=predicts, fill_value=0.0)
    matches = np.full(shape=predicts, fill_value=None)
    for _ in range(min(predicts, trues)):
        pred_mxs = iou.max(axis=1)
        greedy_pred_isnt = pred_mxs.argmax()
        greedy_true_isnt = iou[greedy_pred_isnt].argmax()
        greedy_iou = iou[greedy_pred_isnt, greedy_true_isnt].max()

        # Apaga virtualmente as instancias prevista e verdadeira
        iou[greedy_pred_isnt, :] = -1
        iou[:, greedy_true_isnt] = -1

        matches_iou[greedy_pred_isnt] = greedy_iou
        matches[greedy_pred_isnt]     = greedy_true_isnt

    return matches_iou, matches


def test_limiares(iou, limiares, predicts, trues):
    matches_iou, _ = greedy_match(iou, predicts, trues)

    true_positives = []
    false_positves = []
    false_negatives = []
    average_precisions = []

    # Calcula TP, FP e FN para cada limiar
    for limiar in limiares:
        tp = (matches_iou >= limiar).sum()
        fp = predicts - tp
        fn = trues - tp
        true_positives.append(tp)
        false_positves.append(fp)
        false_negatives.append(fn)
        average_precisions.append(tp / (tp + fp + fn))

    return true_positives, false_positves, false_negatives, average_precisions


def compute_metrics(iou_matrix, limiares):
    iou = iou_matrix[1:, 1:]
    predicts, trues = iou.shape
    true_positives, false_positves, false_negatives, average_precisions = test_limiares(iou, limiares, predicts, trues)
    mean_average_precision = np.mean(average_precisions)
    abs_count_error = np.abs(predicts - trues)

    return true_positives, false_positves, false_negatives, float(mean_average_precision), float(abs_count_error)