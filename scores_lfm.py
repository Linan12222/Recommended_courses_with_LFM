import heapq
import numpy as np
import math
import copy
#计算项目top_K分数，user_count和item_count都只是一个数字
def topK_scores(test, predict, topk, user_count, item_count):
    #精确率和召回率
    instance_count = 2604
    PrecisionSum = np.zeros(topk+1)
    RecallSum = np.zeros(topk+1)
    NDCGSum = np.zeros(topk+1)#归一化折损累计增益
    DCGbest = np.zeros(topk+1)
    MRRSum = 0#平均倒数排名
    total_test_data_count = 0
    hits_sum = 0
    for k in range(1, topk+1):
        DCGbest[k] = DCGbest[k - 1]
        DCGbest[k] += 1.0 / math.log(k + 1)
    for i in range(user_count):
        user_test = []#记录每一个用户的实际值，每一次都要重新迭代
        user_predict = []#记录每一个用户的预测值，每一次都要重新迭代
        test_data_size = 0#计算测试集中，每个用户推荐了多少课程
        for j in range(item_count):
            if test[i * item_count + j] != 0:
                test_data_size += 1
            user_test.append(test[i * item_count + j])
            user_predict.append(predict[i * item_count + j])
        if test_data_size == 0:
            continue
        else:
            #测试集里有1885个用户
            total_test_data_count += 1
        #找到user_predict中最大的topk个值的索引，max_index
        user_predict_copy = copy.deepcopy(user_predict)
        # 求m个最大的数值及其索引
        max_number = []
        max_index = []
        for i in range(topk):
            number = max(user_predict_copy)
            index = user_predict_copy.index(number)
            user_predict_copy[index] = -1
            max_number.append(number)
            max_index.append(index)
        predict_max_num_index_list = max_index
        #定义参数
        hit_sum = 0
        DCG = np.zeros(topk + 1)
        DCGbest2 = np.zeros(topk + 1)
        for k in range(1, topk + 1):
            DCG[k] = DCG[k - 1]
            item_id = predict_max_num_index_list[k - 1]
            if user_test[item_id] != 0:
                # 如果预测对了
                hit_sum += 1
                DCG[k] += 1 / math.log(k + 1)
            # precision, recall
            prec = float(hit_sum / k)
            rec = float(hit_sum / test_data_size)
            PrecisionSum[k] += float(prec)
            RecallSum[k] += float(rec)
            #DCGbst2的最后一个，应该是DCGbest的最后k个
            if test_data_size >= k:
                DCGbest2[k] = DCGbest[k]
            else:
                DCGbest2[k] = DCGbest2[k-1]
            NDCGSum[k] += DCG[k] / DCGbest2[k]
        hits_sum += hit_sum
        # MRR
        p = 1
        for mrr_iter in predict_max_num_index_list:
            if user_test[mrr_iter] != 0:
                break
            p += 1
        MRRSum += 1 / float(p)
    MRR,HR,Prec,Reca,NDCG  = MRRSum / instance_count,hits_sum/instance_count,PrecisionSum[topk-1] / instance_count,RecallSum[topk - 1] / instance_count,NDCGSum[topk-1] / instance_count
    # print("MRR@{}:   {}".format(topk,MRR))
    # print("HR@{}:   {}".format(topk,float(HR)))
    # print("Prec@{}:  {}".format(topk,Prec))
    # print("Reca@{}:   {}".format(topk, Reca))
    # print("NDCG@{}:  {}".format(topk, NDCG))
    # print(' ')
    return MRR,HR,Prec,Reca,NDCG


