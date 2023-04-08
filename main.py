from numpy import *
import pandas as pd


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    #按列数创造一个类别估计数组，初始值为1
    retArr=ones((shape(dataMatrix)[0],1))
    #第dimen列的值小于/大于threshVal置-1，分类准则
    if threshIneq == 'lt':
        retArr[dataMatrix[:,dimen]<=threshVal] = -1.0
    else:
        retArr[dataMatrix[:,dimen]>threshVal]  = 1.0
    return retArr

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat=mat(classLabels).T
    row,col=shape(dataMatrix)
    numSteps=10.0
    # 存储给定权重向量D时所得到的最佳单层决策树相关信息
    bestStump={}
    # 最佳类别估计值
    bestClasEst=mat(zeros((row,1)))
    #最小错误率
    min_Error=inf
    for i in range(col):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        
        for j in range(-1,int(stepSize)+1):
            for inequal in ['lt','gt']:
                threshVal=rangeMin + float(j)*stepSize
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((row,1)))
                # 预测值和真实值相等时，设置为0
                errArr[predictedVals==labelMat] = 0
                weightedError=D.T*errArr
                if weightedError<min_Error:
                    min_Error=weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['inequal']=inequal
    return bestStump,min_Error,bestClasEst

class LogisticRegression:
    
    def __init__(self) :
        self.coef=None
        self.intercept = None  # 截距
        self._theta = None  # _theta[0]是intercept,_theta[1:]是coef
        self.best={}
        self.min_error=inf
        
    def sigmoid(self,z):
        return 1/(1+exp(-z))
    
    def g_descent(self,x_b,y,D,init_theta,eta=0.001,n_iter=1000,l=8):
        theta=init_theta
        iter=0
        while iter <n_iter:
            
            regularized_term = (l/x_b.shape[0]) * theta                          
            regularized_term [0,0]=0
            yz=multiply(y,(x_b @ theta))
            z=self.sigmoid(yz)
            z0=multiply(D,multiply(z-1,y))
            out = -sum(multiply(D , log(1 / (1 + exp (-yz))))) + .5 * l * sum(dot(theta.T, theta))
            #print("put:",out)
            if(out<self.min_error):
                self.min_error=out
                self.best['theta']=theta
                #print("saved!")
            gradient=(x_b.T) @ z0 / len(x_b) + regularized_term
            theta=theta-eta*gradient
            iter+=1
        return theta
    
    def fit(self,data,label,D):
        data=10*data
        x_b = hstack([ones((data.shape[0], 1)), data]) 
        initial_theta = zeros((58,1)) 
        self._theta = self.g_descent(x_b, label, D,initial_theta)
        self._theta=self.best['theta']
        self.intercept = self._theta[0]
        self.coef = self._theta[1:]
        return self
    
    def pre_proba(self,data):
        x_b = hstack([ones((data.shape[0], 1)), data])
        
        return self.sigmoid(x_b.dot(self._theta))
    
    def lrTrain(self,data,label,D):
        m=data.shape[0]
        data=data*10
        prob=self.pre_proba(data)
        pre=where(prob < 0.5, -1, 1)
        errArr=mat(ones((m,1)))
        errArr[pre==label]=0
        we=D.T*errArr
        return self.best,we,pre

def fileToMatrix(data_path,targets_path,base=0):
    #读取数据集
    df = pd.read_csv(data_path,header=None)
    dataMat=df.values
    #读取label集
    lf = pd.read_csv(targets_path,header=None)
    labelMat=(lf.values).T
    row,col=shape(labelMat)
    for i in range(col):
        if(labelMat[0,i]==0):
            labelMat[0,i]=-1.0
        else:
            labelMat[0,i]=1.0
    return dataMat,labelMat

class Adaboost():
    def __init__(self, base=1):
        '''
        :param base: 基分类器编号 0 代表对数几率回归 1 代表决策树桩
        在此函数中对模型相关参数进行初始化，如后续还需要使用参数请赋初始值
        '''
        self.base=base
        self.iter=[1,5,10,100]
        self.bestClassifierArr=[]
    
    def adaClassify(self,dataToClass,classifierArr):
            dataMatrix = mat(dataToClass)
            m=shape(dataMatrix)[0]
            aggClassEst=mat(zeros((m,1)))
            for i in range(len(classifierArr)):
                # 对每个分类器得到一个类别估计值
                if(self.base==1):
                    classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['inequal'])
                else:
                    x_b = hstack([ones((dataMatrix.shape[0], 1)), dataMatrix])
                    proba = 1/(1+exp(-(x_b.dot(classifierArr[i]['theta']))))
                    classEst=where(proba < 0.5, -1, 1)
                aggClassEst += classifierArr[i]['alpha']*classEst
            return sign(aggClassEst)
        
    def fit(self, x_file, y_file):
        '''
        在此函数中训练模型
        :param x_file:训练数据(data.csv)
        :param y_file:训练数据标记(targets.csv)
        '''
        def adaboostTrain(dataArr,classLabels,iter):
            weakClassArr=[]
            m=shape(dataArr)[0]
            # D为每个数据点的权重，每个数据点的权重都会被初始化为1/m
            D = mat(ones((m,1))/m)
            # 记录每个数据点的类别估计累计值
            aggClassEst = mat(zeros((m,1)))
            
            for i in range(iter):
                # 构建一个最佳单层决策树
                if(self.base==1):
                    best,error,classEst = buildStump(dataArr,classLabels,D)
                else:
                    #print("====Adaboost===")
                    lr=LogisticRegression()
                    lr=lr.fit(dataArr,classLabels,D)
                    best,error,classEst=lr.lrTrain(dataArr,classLabels,D)
                    
                alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
                best['alpha']=alpha
                
                # 将最佳单层存储到单层数组中
                weakClassArr.append(best)
                
                # 更新数据样本权值D
                if self.base==1:
                    expt=multiply(-1*alpha*mat(classLabels).T,classEst)
                else:   
                    expt=multiply(-1*alpha*mat(classLabels),classEst)
                D=multiply(D,exp(expt))
                D=D/D.sum()
                
                # 更新累计类别估计值
                aggClassEst +=alpha*classEst
                #print ("aggClassEst: ",aggClassEst.T)
                # 计算错误率
                aggErrors = multiply(sign(aggClassEst)!= mat(classLabels).T,ones((m,1)))
                errorRate = aggErrors.sum()/m
                #print ("errorRate: ",errorRate)
                #print("===========AdaBoost iteration end============")
                if errorRate==0.0:
                    break
            return weakClassArr
        
        #十折
        testRate = 0.1
        dataMat,labelMat=fileToMatrix(x_file, y_file)
        m = dataMat.shape[0]
        labelMat=labelMat.T
        numTestVecs = int(m * testRate)
        
        maxerror=inf
        print("======开始十折验证=======")
        
        for i in range(len(self.iter)):
            print("十折验证过程: ", self.iter[i])
            all = 0
            bestclass=[]
            for k in range(1,11):
                t_test = dataMat[0:numTestVecs]
                #横
                p_test = (labelMat[0:numTestVecs]).T
                xtrain =  dataMat[numTestVecs:m]
                #base=0,不要T
                ytrain = labelMat[numTestVecs:m]
                if self.base==1:
                    ytrain=ytrain.T
                errorCount = 0
                classfierArr=adaboostTrain(xtrain,ytrain,self.iter[i])
                pre10=self.adaClassify(t_test,classfierArr)
                errArr = mat(ones((len(t_test),1)))
                errorCount = errArr[pre10!=mat(p_test).T].sum()
                #----------将第几折的数据拿出来，放回到normMat的前面
                b = dataMat[numTestVecs*(k-1):numTestVecs*k]
                dataMat[0:numTestVecs] = b
                dataMat[numTestVecs*(k-1):numTestVecs*k] = t_test
                #----------将第几折类别拿出来，放回到datingLabels的前面
                c = labelMat[numTestVecs*(k-1):numTestVecs*k]
                labelMat[0:numTestVecs] = c
                labelMat[numTestVecs*(k-1):numTestVecs*k] = p_test.T
                #------------------------------------------------------------------
                errorRate = errorCount/float(numTestVecs)
                all = all + errorRate
                print("第%d折分类的错误率为%f" % (k,(errorCount/float(numTestVecs))))
                
                if(errorRate<maxerror):
                    maxerror=errorRate
                    bestclass=classfierArr
                self.bestClassifierArr.append(bestclass)
                #写入csv
                pre10=where(pre10<0,0,1)
                path='data_examples/experiments/base'+str(self.iter[i])+'_fold'+str(k)+'.csv'
                preframe=pd.DataFrame(pre10)
                preframe.index = arange(1+numTestVecs*(k-1), numTestVecs*k+1)
                preframe.to_csv(path,header=None)
                
            #获得平均错误率
            print("平均正确率为%f" % (1-(all/10)))
        
    def predict(self, x_file):
        '''
        :param x_file:测试集文件夹(后缀为csv)
        :return: 训练模型对测试集的预测标记
        '''
        preArr=[]
        df = pd.read_csv(x_file,header=None)
        dataMat=df.values
        for i in (len(self.iter)):
            pre=self.adaClassify(dataMat,self.bestClassifierArr[i])
            preArr[i]=where(pre<0,0,1)
        
            
if __name__=='__main__':
    adaboost=Adaboost()
    adaboost.fit('data_examples/data.csv',
            'data_examples/targets.csv')