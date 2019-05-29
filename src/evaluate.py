"""
Created on March 5, 2019
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    原：Measures: Hit Ratio and NDCG
    修改为：Measures：Precision 、Recall 、MAP 、NDCG



"""
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits = 0
    test_length = 0
    recset_length = 0
    ap = 0
    ndcg = 0
    precision_K=0
    recall_K=0

    for idx in range(len(_testRatings)):
        hitsu, len_ranklist, len_test, ndcgu, apu,= eval_one_rating(idx)

        hits += hitsu
        recset_length += len_ranklist
        test_length+= len_test
        ap += apu
        ndcg += ndcgu

    precision = hits / (1.0 * recset_length)
    recall= hits / (1.0 * test_length)
    ap = ap * 1.0 / len(_testRatings)
    ndcg = 1.0 * ndcg / len(_testRatings)
    # hitratio_K=hitratio* 1.0/len(_testRatings)
    return precision, recall, ap, ndcg


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    test = rating[1]

    for poi in test:
        items.append(poi)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    # items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score.keys(), key=map_item_score.get)
    # if len(ranklist) == 0:
    #     print(idx)
    hitsu = 0
    for test_poi in test:
        if test_poi in ranklist:
            hitsu += 1
    apu = AP(ranklist, test)
    ndcgu = ndcg(ranklist, test, _K)

    return hitsu,len(ranklist), len(test),ndcgu,apu,


def AP(ranked_list, ground_truth):
    """Compute the average precision (AP) of a list of ranked items

    """
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0
# numpy.reciprocal() 函数返回参数逐元素的倒数


def ndcg(ranked_list, ground_truth,topk):
    dcg = 0
    pred_rel = []
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            pred_rel.append(topk)
        else:
            pred_rel.append(0)
        topk -= 1
    for (index, rel) in enumerate(pred_rel):
        dcg += (rel * np.reciprocal(np.log2(index + 2)))

    idcg = 0
    for (index, rel) in enumerate(sorted(pred_rel, reverse=True)):
        idcg += (rel * np.reciprocal(np.log2(index + 2)))
    if idcg == 0:
        ndcgu=0
    else:
        ndcgu=dcg/idcg
    if math.isnan(ndcgu):
        ndcgu=0
    return ndcgu
