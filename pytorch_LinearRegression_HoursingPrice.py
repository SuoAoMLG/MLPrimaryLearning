import torch
from torch.autograd import Variable
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import numpy
import csv
import pandas as pd
import torch.utils.data as Data
from sklearn.decomposition import PCA 
LR = 0.05                       #learning rate 学习率
times=100
#这个类用来导入数据
class DataInit:
    def __init__(self,DataSource,author):                   #初始函数用来初始换类里面的固有的属性和方法                                               #self参数似乎是伴随整个类的指向自己参数
        self.DataSource=DataSource
        self.author=author
                                                            #定义在类里面的方法（函数）"E:\\MachineLearning\\pytorch\\train.csv"
    def DataPrepare(self):
            self.saleprice = pd.read_csv(self.DataSource, usecols=['SalePrice'])
            self.input = pd.read_csv(self.DataSource)

#数据导入

datatrain = DataInit("E:\\MachineLearning\\homework\\house-prices-advanced-regression-techniques\\train.csv",'jing')
datatrain.DataPrepare()
x=datatrain.input
y=datatrain.saleprice

#我们把数据预处理封装成一个类，这里我参考了kaggle上面的一个大佬的数据准备，具体见https://blog.csdn.net/wydyttxs/article/details/79680814
class data_prepare:
    def __init__(self,x,author):
        self.x = x
        self.author=author
    def data_replenish(self):
        #一开始是进行数据填充，处理残缺数据，英文是原注释
        #Let's first imput the missing values of LotFrontage based on the median of LotArea and Neighborhood. 
        # Since LotArea is a continuous feature, We use qcut to divide it into 10 parts.
        self.x.groupby(['Neighborhood'])[['LotFrontage']].agg(['mean','median','count'])
        self.x["LotAreaCut"] = pd.qcut(self.x.LotArea,10)
        self.x.groupby(['LotAreaCut'])[['LotFrontage']].agg(['mean','median','count'])
        self.x['LotFrontage']=self.x.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x1: x1.fillna(x1.median()))
        self.x['LotFrontage']=self.x.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x1: x1.fillna(x1.median()))
        #这一部分采取了分类加中位数填充

        #Then we filling in other missing values according to data_description.
        cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
        for col in cols1:
            self.x[col].fillna("None", inplace=True)
        cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
        for col in cols:
            self.x[col].fillna(0, inplace=True)
        cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
        for col in cols2:
            self.x[col].fillna(self.x[col].mode()[0], inplace=True)
        #这一部分是直接规定的填充

    #接下来我们对数据中的部分进行数据映射
    def map_values(self):
        self.x["oMSSubClass"] = self.x.MSSubClass.map({'180':1, 
                                        '30':2, '45':2, 
                                        '190':3, '50':3, '90':3, 
                                        '85':4, '40':4, '160':4, 
                                        '70':5, '20':5, '75':5, '80':5, '150':5,
                                        '120': 6, '60':6})
    
        self.x["oMSZoning"] =self.x.MSZoning.map({'C (all)':1, 'RH':2, 'RM':2, 'RL':3, 'FV':4})
    
        self.x["oNeighborhood"] = self.x.Neighborhood.map({'MeadowV':1,
                                               'IDOTRR':2, 'BrDale':2,
                                               'OldTown':3, 'Edwards':3, 'BrkSide':3,
                                               'Sawyer':4, 'Blueste':4, 'SWISU':4, 'NAmes':4,
                                               'NPkVill':5, 'Mitchel':5,
                                               'SawyerW':6, 'Gilbert':6, 'NWAmes':6,
                                               'Blmngtn':7, 'CollgCr':7, 'ClearCr':7, 'Crawfor':7,
                                               'Veenker':8, 'Somerst':8, 'Timber':8,
                                               'StoneBr':9,
                                               'NoRidge':10, 'NridgHt':10})
    
        self.x["oCondition1"] = self.x.Condition1.map({'Artery':1,
                                           'Feedr':2, 'RRAe':2,
                                           'Norm':3, 'RRAn':3,
                                           'PosN':4, 'RRNe':4,
                                           'PosA':5 ,'RRNn':5})
    
        self.x["oBldgType"] = self.x.BldgType.map({'2fmCon':1, 'Duplex':1, 'Twnhs':1, '1Fam':2, 'TwnhsE':2})
    
        self.x["oHouseStyle"] = self.x.HouseStyle.map({'1.5Unf':1, 
                                           '1.5Fin':2, '2.5Unf':2, 'SFoyer':2, 
                                           '1Story':3, 'SLvl':3,
                                           '2Story':4, '2.5Fin':4})
    
        self.x["oExterior1st"] = self.x.Exterior1st.map({'BrkComm':1,
                                             'AsphShn':2, 'CBlock':2, 'AsbShng':2,
                                             'WdShing':3, 'Wd Sdng':3, 'MetalSd':3, 'Stucco':3, 'HdBoard':3,
                                             'BrkFace':4, 'Plywood':4,
                                             'VinylSd':5,
                                             'CemntBd':6,
                                             'Stone':7, 'ImStucc':7})
    
        self.x["oMasVnrType"] = self.x.MasVnrType.map({'BrkCmn':1, 'None':1, 'BrkFace':2, 'Stone':3})
    
        self.x["oExterQual"] = self.x.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
        self.x["oFoundation"] = self.x.Foundation.map({'Slab':1, 
                                           'BrkTil':2, 'CBlock':2, 'Stone':2,
                                           'Wood':3, 'PConc':4})
    
        self.x["oBsmtQual"] = self.x.BsmtQual.map({'Fa':2, 'None':1, 'TA':3, 'Gd':4, 'Ex':5})
    
        self.x["oBsmtExposure"] = self.x.BsmtExposure.map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})
    
        self.x["oHeating"] = self.x.Heating.map({'Floor':1, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':5})
    
        self.x["oHeatingQC"] = self.x.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
        self.x["oKitchenQual"] = self.x.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
    
        self.x["oFunctional"] = self.x.Functional.map({'Maj2':1, 'Maj1':2, 'Min1':2, 'Min2':2, 'Mod':2, 'Sev':2, 'Typ':3})
    
        self.x["oFireplaceQu"] = self.x.FireplaceQu.map({'None':1, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    
        self.x["oGarageType"] = self.x.GarageType.map({'CarPort':1, 'None':1,
                                           'Detchd':2,
                                           '2Types':3, 'Basment':3,
                                           'Attchd':4, 'BuiltIn':5})
    
        self.x["oGarageFinish"] = self.x.GarageFinish.map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})
    
        self.x["oPavedDrive"] = self.x.PavedDrive.map({'N':1, 'P':2, 'Y':3})
    
        self.x["oSaleType"] = self.x.SaleType.map({'COD':1, 'ConLD':1, 'ConLI':1, 'ConLw':1, 'Oth':1, 'WD':1,
                                       'CWD':2, 'Con':3, 'New':3})
    
        self.x["oSaleCondition"] = self.x.SaleCondition.map({'AdjLand':1, 'Abnorml':2, 'Alloca':2, 'Family':2, 'Normal':3, 'Partial':4})            
        self.x.drop('LotAreaCut',axis=1,inplace=True)  #LotAreaCut 是分类求中位数从而来补全数据产生的附加列，这里它不参与神经网络，我们删掉这一列
    
    def romovedata(self):                     
        self.x.drop('SalePrice',axis=1, inplace=True) #由于原来数据全部被导入，这里要删掉'SalePrice'列，包括以后的数据导入，这里要注意
    #接下来是数据重组，这里我直接选取那个大佬的数据重组选择，实际上有好多种组合方法，大佬的代码里面还有更复杂的主成分分析和选取部分，
    #这里我直接提取了其中一部分
    def transform(self):
           
            self.x["TotalHouse"] = self.x["TotalBsmtSF"] + self.x["1stFlrSF"] + self.x["2ndFlrSF"]   
            self.x["TotalArea"] = self.x["TotalBsmtSF"] + self.x["1stFlrSF"] + self.x["2ndFlrSF"] + self.x["GarageArea"]
            
            self.x["+_TotalHouse_OverallQual"] = self.x["TotalHouse"] * self.x["OverallQual"]
            self.x["+_GrLivArea_OverallQual"] = self.x["GrLivArea"] * self.x["OverallQual"]
            self.x["+_oMSZoning_TotalHouse"] = self.x["oMSZoning"] * self.x["TotalHouse"]
            self.x["+_oMSZoning_OverallQual"] = self.x["oMSZoning"] + self.x["OverallQual"]
            self.x["+_oMSZoning_YearBuilt"] = self.x["oMSZoning"] + self.x["YearBuilt"]
            self.x["+_oNeighborhood_TotalHouse"] = self.x["oNeighborhood"] * self.x["TotalHouse"]
            self.x["+_oNeighborhood_OverallQual"] = self.x["oNeighborhood"] + self.x["OverallQual"]
            self.x["+_oNeighborhood_YearBuilt"] = self.x["oNeighborhood"] + self.x["YearBuilt"]
            self.x["+_BsmtFinSF1_OverallQual"] = self.x["BsmtFinSF1"] * self.x["OverallQual"]
            
            self.x["-_oFunctional_TotalHouse"] = self.x["oFunctional"] * self.x["TotalHouse"]
            self.x["-_oFunctional_OverallQual"] = self.x["oFunctional"] + self.x["OverallQual"]
            self.x["-_LotArea_OverallQual"] = self.x["LotArea"] * self.x["OverallQual"]
            self.x["-_TotalHouse_LotArea"] = self.x["TotalHouse"] + self.x["LotArea"]
            self.x["-_oCondition1_TotalHouse"] = self.x["oCondition1"] * self.x["TotalHouse"]
            self.x["-_oCondition1_OverallQual"] = self.x["oCondition1"] + self.x["OverallQual"]
            
           
            self.x["Bsmt"] = self.x["BsmtFinSF1"] + self.x["BsmtFinSF2"] + self.x["BsmtUnfSF"]
            self.x["Rooms"] = self.x["FullBath"]+self.x["TotRmsAbvGrd"]
            self.x["PorchArea"] = self.x["OpenPorchSF"]+self.x["EnclosedPorch"]+self.x["3SsnPorch"]+self.x["ScreenPorch"]
            self.x["TotalPlace"] = self.x["TotalBsmtSF"] + self.x["1stFlrSF"] + self.x["2ndFlrSF"] + self.x["GarageArea"] + self.x["OpenPorchSF"]+self.x["EnclosedPorch"]+self.x["3SsnPorch"]+self.x["ScreenPorch"]

            self.x=self.x[["TotalHouse","TotalArea","+_TotalHouse_OverallQual","+_GrLivArea_OverallQual","+_oMSZoning_TotalHouse",
            "+_oMSZoning_OverallQual","+_oMSZoning_YearBuilt","+_oNeighborhood_TotalHouse","+_oNeighborhood_OverallQual",
            "+_oNeighborhood_YearBuilt","+_BsmtFinSF1_OverallQual","-_oFunctional_TotalHouse","-_oFunctional_OverallQual",
            "-_LotArea_OverallQual","-_TotalHouse_LotArea","-_oCondition1_TotalHouse","-_oCondition1_OverallQual",
            "Bsmt","Rooms","PorchArea","TotalPlace"
            ]]
            #重新和成的x输入数据
    #最后是PCA 处理数据，这里由于之前已经进行过了变量组合（feature compile），所以不需要再过分的降维处理了
    def datapca(self):
            pca=PCA(n_components=20)
            self.x=pca.fit_transform(self.x)
            self.x=torch.from_numpy(self.x).type(torch.FloatTensor)
            

#对学习数据预处理
dataPre = data_prepare(x,'jing')
dataPre.data_replenish()
dataPre.map_values()
dataPre.romovedata()
dataPre.transform()
dataPre.datapca()
x=dataPre.x
y=y.as_matrix().reshape((-1,1))
y=torch.from_numpy(y).type(torch.FloatTensor)
#print(x)#[1460 rows x 20 columns]
#print(y)#[1460 rows x 1 columns]


#这里才开始神经网络，最基础的线性回归
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

#这是定义神经网络的输入features，隐藏层数，输出数量的Net类
net = Net(20,10,1)
print(net)

#optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.1, 0.11))#优化器adam，参数可调
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(times):                    #迭代100次（参数可调）
    prediction = net(x)
    loss = loss_func(prediction,y)
    optimizer.zero_grad()               #将最原始的梯度清零
    loss.backward()                     #直接计算梯度（封装好的函数），执行反向回归算法
    optimizer.step()                    #计算梯度，然后开始梯度下降
    

#导入test数据
x1= pd.read_csv("E:\\MachineLearning\\homework\\house-prices-advanced-regression-techniques\\test.csv")
x2 = pd.read_csv("E:\\MachineLearning\\homework\\house-prices-advanced-regression-techniques\\test.csv", usecols=['Id'])
x2 = x2.as_matrix().squeeze()
dataPre1 = data_prepare(x1,'jing')
dataPre1.data_replenish()
dataPre1.map_values()
#dataPre1.romovedata()   这一句注意在导入的数据里面没有saleprice，所以不要操作
dataPre1.transform()
dataPre1.datapca()
x1=dataPre1.x
#print(x1)
prediction1=net(x1)
print(prediction1.data.numpy())

#打印预测数据
datatest_pred = list(zip(x2,prediction1.data.numpy().squeeze()))
datatest_pred_dataframe = pd.DataFrame(data = datatest_pred ,columns=['Id','SalePrice'])
#print(datatest_pred_dataframe)
datatest_pred_dataframe.to_csv("E:\\MachineLearning\\homework\\house-prices-advanced-regression-techniques\\test_pred.csv", index=False, header=True )
