import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import scores_lfm
'''
R   M*N的评分矩阵
P   初始化用户特征矩阵M*K
Q   初始化用户特征矩阵N*K
K   隐特征向量维度
max_iter 最大迭代次数
alpha 步长
lamda 正则化系数
'''
K = 10
max_iter = 1500
alpha = 0.0002
lamda = 0.004
def LFM_grad_desc(R,K,max_iter,alpha,lamda):
    M = len(R)
    N = len(R[0])
    #生成随机初始值
    P = np.random.rand(M,K)
    Q = np.random.rand(N,K)
    Q = Q.T
    #开始迭代
    for step in range(max_iter):
        for u in range(M):
            for i in range(N):
                #对于每一个大于0的评分，求出预测评分误差，对于等于0的评分，不考虑其误差
                if R[u][i]>0:
                    eui = np.dot(P[u,:],Q[:,i])-R[u][i]
                    #代入公式，按照梯度下降算法更新Pu，Qi
                    for k in range(K):
                        P[u][k] = P[u][k] - alpha*2*(eui*Q[k][i]+lamda*P[u][k])
                        Q[k][i] = Q[k][i] - alpha*2*(eui*P[u][k]+lamda*Q[k][i])
        #u、i遍历完成，所有特征向量更新完成，可以得到P、Q, 可以计算预测评分矩阵
        predR=np.dot(P,Q)
        #计算当前损失函数
        loss= 0
        for u in range (M):
            for i in range (N):
                if R[u][i] > 0:
                    loss += (np.dot(P[u,:],Q[:,i])-R[u][i])**2
                    #加上正则化项
                    for k in range(K) :
                        loss+=lamda*(P[u][k]**2+Q[k][i]**2)
        if step%300==0:
            print("第{}次迭代,loss为{}".format(step,loss))

    return P,Q.T,loss


#lfmtrain1是有序的no disrupt，lfmtrain2是无序的disrupt
names = ['lfmtrain1','lfmtrain2']
sheet_names= ['no disrupt','disrupt']
writer = pd.ExcelWriter('lfm_index.xlsx')
for name in names:
    R = np.zeros([2322,685])
    with open(name + '.txt') as f:
        for line in f:
            a,b,c = line.split(' ')
            R[int(a)-1][int(b)-1] = int(c)
    indexs = pd.DataFrame(np.zeros([10, 21]))
    for _ in range(1,11):
        print('第{}次运行'.format(str(_)))
        P,Q,loss = LFM_grad_desc(R,K,max_iter,alpha,lamda)
        predR = P.dot(Q.T)
        #用户数量，项目数量
        user_count = 2322
        item_count = 685
        #从excel文件中整理出Predict_,最终结果为(1590570,)的ndarray
        predict_ = predR
        predict_ = predict_.reshape((1590570,))
        #从txt
        # 从txt文件中整理出test,最终结果为(1590570,)的ndarray
        test = np.zeros((2322,685))
        test_auc = np.zeros((2322,685))
        with open('lfmtest'+name[-1]+'.txt') as f:
            for line in f:
                a,b,c = line.split(' ')
                test[int(a)-1,int(b)-1] = float(c)
                test_auc[int(a)-1,int(b)-1] = 1
        test = test.reshape((1590570,))
        test_auc = test_auc.reshape((1590570,))
        auc_score = roc_auc_score(test_auc, predict_)
        print('AUC:     {}'.format(auc_score))
        # Top-K evaluation
        topks = [1,5,10,20]
        MRRS,HRS,Precs,Recas,NDCGs = [auc_score],[],[],[],[]
        for topk in topks:
            MRR,HR,Prec,Reca,NDCG = scores_lfm.topK_scores(test, predict_, topk, user_count, item_count)
            MRRS.append(MRR),HRS.append(HR),Precs.append(Prec),Recas.append(Reca),NDCGs.append(NDCG)
        index = pd.DataFrame(MRRS+HRS+Precs+Recas+NDCGs).T
        indexs.loc[_-1, :] = index.loc[0, :]
    indexs.columns = ['AUC', 'MRR@1', 'MRR@5', 'MRR@10', 'MRR@20', 'HR@1', 'HR@5', 'HR@10', 'HR@20', 'Prec@1', 'Prec@5',
                             'Prec@10', 'Prec@20', 'Reca@1', 'Reca@5', 'Reca@10', 'Reca@20', 'NDCG@1', 'NDCG@5', 'NDCG@10',
                             'NDCG@20']
    indexs.loc[10] = indexs.apply(lambda x: x.mean())
    indexs.index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'mean']
    indexs.to_excel(writer,sheet_name=name)
writer.close()