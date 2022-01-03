import numpy as np
import math

def CalculateARI(n, predClassCount, refClassCount, classrefmat):
    rowsum = np.zeros(predClassCount, dtype = 'float32')
    colsum = np.zeros(refClassCount, dtype = 'float32')
    rowsumC2 = np.zeros(predClassCount, dtype = 'float32')
    colsumC2 = np.zeros(refClassCount, dtype = 'float32')
    classrefmatC2 = np.zeros([predClassCount, refClassCount], dtype = 'float64')
    nC2 = n * (n - 1) / 2
    sumclassrefmatC2 = 0
    sumrowsumC2 = 0
    sumcolsumC2 = 0
    for predClass in range(predClassCount):
        for refClass in range(refClassCount):
            rowsum[predClass] += classrefmat[predClass][refClass]
            colsum[refClass] += classrefmat[predClass][refClass]
            classrefmatC2[predClass][refClass] = classrefmat[predClass][refClass] * \
                        (classrefmat[predClass][refClass] - 1) / 2
            sumclassrefmatC2 += classrefmatC2[predClass][refClass]
    for predClass in range(predClassCount):
        rowsumC2[predClass] = rowsum[predClass] * (rowsum[predClass] - 1) / 2
        sumrowsumC2 += rowsumC2[predClass]
    for refClass in range(refClassCount):
        colsumC2[refClass] = colsum[refClass] * (colsum[refClass] - 1) / 2
        sumcolsumC2 += colsumC2[refClass]
    prod_comb = sumrowsumC2 * (sumcolsumC2 / nC2)
    mean_comb = (sumrowsumC2 + sumcolsumC2) / 2
    ARI_numerator = (sumclassrefmatC2 - prod_comb)
    ARI_denominator = (mean_comb - (sumrowsumC2 * (sumcolsumC2 / nC2)))
    ARI = ARI_numerator / ARI_denominator
    return ARI

def CalculateNMI(n, predClassCount, refClassCount, classrefmat):
    H_predClass = 0
    H_refClass = 0
    MI_predClass_refClass = 0
    rowsum = np.zeros(predClassCount, dtype = 'float32')
    colsum = np.zeros(refClassCount, dtype = 'float32')
    classrefmatC2 = np.zeros([predClassCount, refClassCount], dtype = 'float64')
    for predClass in range(predClassCount):
        for refClass in range(refClassCount):
            rowsum[predClass] += classrefmat[predClass][refClass]
            colsum[refClass] += classrefmat[predClass][refClass]
            classrefmatC2[predClass][refClass] = classrefmat[predClass][refClass] * (classrefmat[predClass][refClass] - 1) / 2
    for predClass in range(predClassCount):
        if(rowsum[predClass] > 0):
            H_predClass += (rowsum[predClass] / n) * math.log(rowsum[predClass] / n)
    H_predClass = H_predClass * (-1)

    for refClass in range(refClassCount):
        if(colsum[refClass] > 0):
            H_refClass += (colsum[refClass] / n) * math.log(colsum[refClass] / n)
    H_refClass = H_refClass * (-1)

    for predClass in range(predClassCount):
        for refClass in range(refClassCount):
            if(classrefmat[predClass][refClass] != 0):
                P_predClass_refClass = classrefmat[predClass][refClass] / n
                log_PpredClassRefClassByPpredClassPrefClass = math.log((classrefmat[predClass][refClass] * n) / (rowsum[predClass] * colsum[refClass]))
                MI_predClass_refClass += P_predClass_refClass * log_PpredClassRefClassByPpredClassPrefClass
    NMI = MI_predClass_refClass / math.sqrt(H_predClass * H_refClass)
    return NMI

def clustering_eval(predRefClsCount, classLabels, refLabels):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    numInstances = len(classLabels)
    predClassCount = predRefClsCount[0]
    refClassCount = predRefClsCount[1]
    classrefmat = np.zeros([predClassCount, refClassCount], dtype = int)
    majorityClass = np.zeros(refClassCount, dtype = int)

    for i in range(len(classLabels)):
        pred = classLabels[i]
        ref = refLabels[i]
        classrefmat[pred][ref] += 1
    for predClass in range(predClassCount):
        fpinrow = 0
        for k in range(refClassCount):
            fpinrow += classrefmat[predClass][k]
        for refClass in range(refClassCount):
            tp += ((classrefmat[predClass][refClass]-1) * classrefmat[predClass][refClass]) / 2
            fp += classrefmat[predClass][refClass] * (fpinrow - classrefmat[predClass][refClass])
    fp = fp / 2

    for refClass in range(refClassCount):
        fnincol = 0
        for k in range(predClassCount):
            fnincol += classrefmat[k][refClass]
        for predClass in range(predClassCount):
            fn += classrefmat[predClass][refClass] * (fnincol - classrefmat[predClass][refClass])
            if (majorityClass[refClass] < classrefmat[predClass][refClass]):
                majorityClass[refClass] = classrefmat[predClass][refClass]
    fn = fn / 2
    tn = numInstances * ((numInstances - 1) / 2) - tp - fp - fn

    sumMajorityClass = 0
    for i in range(refClassCount):
        sumMajorityClass += majorityClass[i]
    purity = sumMajorityClass / (float)(numInstances)
    RI = (tp + tn) / (float)(tp + fp + tn + fn)
    prec = (tp) / (float)(tp + fp)
    recall = (tp) / (float)(tp + fn)
    fscore = 2 * prec * recall / (float)(prec + recall)
    jac = tp / (float)(fp + fn + tp)
    ARI = CalculateARI(numInstances, predClassCount, refClassCount, classrefmat)
    NMI = CalculateNMI(numInstances, predClassCount, refClassCount, classrefmat)
    print("F-score\tARI\tNMI\t%0.4f\t%0.4f\t%0.4f"%(fscore, ARI, NMI))
