from __future__ import division, print_function
__author__ = 'Mahboubeh'
##It is the last version of link prediction for our idea
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.linear_model import Ridge
import numpy as np
import networkx as nx
from sklearn.cross_validation import KFold
import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import csv


xrange=range

##compute out degree u (out degree positive and negative)
def OutDegree(N,Matrix,u):
    Poetive=0
    Negative=0
    Degree=[]
    for vertice in N:
        if (Matrix[u][vertice]<-1):
             Negative+=Matrix[u][vertice]
        if (Matrix[u][vertice]>1):
            Poetive+=Matrix[u][vertice]
    Degree.append(Poetive)
    Degree.append(Negative)
    return Degree
###############################
##Compute indegree V(indegree positive and negative)
def InDegree(N,Matrix,v):
    Poetive = 0
    Negative = 0
    Degree = []
    for vertice in N:
       if (Matrix[vertice][v] < 0):
         Negative+=Matrix[vertice][v]
       if (Matrix[vertice][v] > 0):
            Poetive+=Matrix[vertice][v]
    Degree.append(Poetive)
    Degree.append(Negative)
    return Degree
#################################
def successIn(N, Matrix, v):
    Degree = []
    success =0
    fail = 0
    for vertice in N:
        if Matrix[vertice][v]>0:
            success+=Matrix[vertice][v]
        else:
            fail += Matrix[vertice][v]
    Degree.append(success)
    Degree.append(fail)
    return Degree
#################################
def successout(N,Matrix,u):
    Degree=[]
    success =0
    fail = 0
    for vertice in N:
        if Matrix[u][vertice]>0:
            success+=Matrix[u][vertice]
        else:
            fail+=Matrix[u][vertice]
    Degree.append(success)
    Degree.append(fail)
    return Degree
###############################
target_names = ['class 0', 'class 1']


nodeA = []
nodeB = []
sign = []
game = []
weight = []
TeamH = []
TeamA = []
G = nx.DiGraph()
with open('CleanData.csv') as csvfile:
   spamreader = csv.DictReader(csvfile)
   for row in spamreader:
      g = []
      nodeA.append(row['Home'])
      nodeB.append(row['Away'])
      sign.append(row['Result'])
      # game.append(row['GameID'])
      # weight.append(row['Weight'])
      # TeamH.append(row['TeamH'])
      # TeamA.append(row['TeamA'])

node=[]
node.append(nodeA[0])
for x in xrange(1,len(nodeA)):
    if (nodeA[x] not in node):
        node.append(nodeA[x])

for x in xrange(0,len(nodeB)):
    if(nodeB[x] not in node):
      node.append(nodeB[x])
M= len(node)
print("number of node %s" % M)

###########################


edge = [[0 for x in range(2)] for y in range(len(nodeA))]
S = [[0 for x in range(3)] for y in range(len(nodeA))]

counF = 0
countB = 0
index = 0
#Direction is from home to guest
#sign shows winner of the game
for i in xrange(0, len(nodeA)):
    temp = []
    temp1 = []
    U = node.index(nodeA[i])
    V = node.index(nodeB[i])
    temp.append(U)
    temp.append(V)
    # temp1.append(game[i])
    # temp1.append(TeamH[i])
    # temp1.append(TeamA[i])
    # temp1.append(weight[i])
    edge[index] = temp
    if float(sign[i])< 0 :
        temp1.append(-1)
        counF+=1
    else:
        temp1.append(1)
        countB+=1

    S[index] = temp1
    index += 1

print("number of edge:", index)
print("Positive: ", counF)
print("Negative: ", countB)
documents = edge

# ############################
n_folds = 10
skf = KFold(n=len(edge), n_folds=n_folds, shuffle=True)
main_f = 0
main_p = 0
main_r = 0
fscore = 0
print("Divide data to two categorize:")
for train_index, test_index in skf:
    collections = np.array(documents)
    labels = np.array(S)
    x_train, x_test = collections[train_index], collections[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
##############################################################
###creat network based on train
    size=len(x_train)
    G = nx.Graph()
    Adjancy = np.zeros((M,M),dtype=np.float) #we need this matrix for indegree and outdegree
    ##creat network base on train data and creat adjancy matrix
    print("Start creating Networks")
    for i in xrange(0, size):
        u = x_train[i][0]
        v = x_train[i][1]
        Adjancy[u][v] += float(y_train[i])
        # G.add_edge(u, v)
        G.add_path([u, v])
#############################################
 ##find all clique in the network train
    print("Feature Matrix for Train")
    Feature_train = np.zeros((size, 16))
    labels_train = np.zeros((size, 1))
# # # # # # ##########################################
    start4 = datetime.datetime.now()
    for ii in xrange(0,size):
        TempF=[]
        u1=x_train[ii][0]
        v1=x_train[ii][1]
        degreeU = G.degree(u1)
        degreeV = G.degree(v1)
        SignUV=Adjancy[u1][v1]
        SignVU=Adjancy[v1][u1]
        Adjancy[u1][v1]=0
        Adjancy[v1][u1]=0
        ##################################################

        CounterUV = 0
        CounterVU = 0
        similar = 0
        similarN = 0
        similarP = 0
        ######Compute Sp in Whole graph(G)
        G.remove_edge(u1, v1) # we should remove edge (u,v) then compute shortest path between u and v.
        # if we don't remove edge (u,v) so all shortest path between u and v will be one
        try:
            CounterUV =nx.shortest_path_length(G, u1, v1)

        except nx.exception.NetworkXNoPath:
            CounterUV =0

        try:
            CounterVU =nx.shortest_path_length(G, v1, u1)

        except nx.exception.NetworkXNoPath:
            CounterVU =0

        ###########################
        x = nx.neighbors(G, u1)
        y = nx.neighbors(G, v1)

        outdegreeU = OutDegree(x, Adjancy, u1)
        sumInU= outdegreeU[0] + outdegreeU[1]
        IndegreeU = InDegree(x, Adjancy, u1)
        sumOutU = IndegreeU[0] + IndegreeU[1]

        IndegreeV = InDegree(y, Adjancy, v1)
        sumInV = IndegreeV[0] + IndegreeV[1]
        outdegreeV = OutDegree(y, Adjancy, v1)
        sumOutV = outdegreeV[0] + outdegreeV[1]

        # TempF.append(sumInU)
        # TempF.append(sumOutU)
        # TempF.append(sumInV)
        # TempF.append(sumOutV)
        SucInU = successIn (x, Adjancy, u1)
        SucOutU = successout(x, Adjancy, u1)
        successU = SucInU[0] + SucOutU[0]
        # TempF.append(successU/degreeU)
        ##########################
        FailInU = SucInU[1] + SucOutU[1]
        # TempF.append(FailInU/degreeU)
        ##########################
        SucInV = successIn(x, Adjancy, u1)
        SucOutV = successout(x, Adjancy, u1)
        successV = SucInV[0] + SucOutV[0]
        # TempF.append(successV/degreeV)
        ##########################
        FailInV = SucInV[1] + SucOutV[1]
        # TempF.append(FailInV/degreeV)
        ##########################
        ##define new measure
        FPP = 0
        FPN = 0
        FNP = 0
        FNN = 0
        FRPP = 0
        FRPN = 0
        FRNP = 0
        FRNN = 0
        IPP = 0
        IPN = 0
        INN = 0
        INP = 0
        OPP = 0
        OPN = 0
        ONP = 0
        ONN = 0

        # FPP1 = 0
        # FPN1 = 0
        # FNP1 = 0
        # FNN1 = 0
        # FRPP1 = 0
        # FRPN1 = 0
        # FRNP1 = 0
        # FRNN1 = 0
        # IPP1 = 0
        # IPN1 = 0
        # INN1 = 0
        # INP1 = 0
        # OPP1 = 0
        # OPN1 = 0
        # ONP1 = 0
        # ONN1 = 0

        if CounterUV != 0:

            FPP = (outdegreeU[0] * IndegreeV[0])/(CounterUV * CounterUV)
            FPN = (outdegreeU[0] * IndegreeV[1]) / (CounterUV * CounterUV)
            FNP = (outdegreeU[1] * IndegreeV[0]) / (CounterUV * CounterUV)
            FNN = (outdegreeU[1] * IndegreeV[1]) / (CounterUV * CounterUV)
            FRPP = (outdegreeV[0] * IndegreeU[0])/(CounterUV * CounterUV)
            FRPN = (outdegreeV[0] * IndegreeU[1])/(CounterUV * CounterUV)
            FRNP = (outdegreeV[1] * IndegreeU[0]) / (CounterUV * CounterUV)
            FRNN = (outdegreeV[1] * IndegreeU[1])/(CounterUV * CounterUV)
            IPP = (IndegreeV[0] * IndegreeU[0])/(CounterUV * CounterUV)
            IPN = (IndegreeV[0] * IndegreeU[1]) / (CounterUV * CounterUV)
            INP = (IndegreeV[1] * IndegreeU[0]) / (CounterUV * CounterUV)
            INN = (IndegreeV[1] * IndegreeU[1]) / (CounterUV * CounterUV)
            OPP = (outdegreeV[0] * outdegreeU[0])/(CounterUV * CounterUV)
            OPN = (outdegreeV[0] * outdegreeU[1]) / (CounterUV * CounterUV)
            ONP = (outdegreeV[1] * outdegreeU[0]) / (CounterUV * CounterUV)
            ONN = (outdegreeV[1] * outdegreeU[1]) / (CounterUV * CounterUV)

        # if CounterVU !=0:
        #     FPP1 = (outdegreeU[0] * IndegreeV[0])/(CounterVU * CounterVU)
        #     FPN1 = (outdegreeU[0] * IndegreeV[1]) / (CounterVU * CounterVU)
        #     FNP1 = (outdegreeU[1] * IndegreeV[0]) / (CounterVU * CounterVU)
        #     FNN1 = (outdegreeU[1] * IndegreeV[1]) / (CounterVU * CounterVU)
        #     FRPP1 = (outdegreeV[0] * IndegreeU[0])/(CounterVU * CounterVU)
        #     FRPN1 = (outdegreeV[0] * IndegreeU[1])/(CounterVU * CounterVU)
        #     FRNP1 = (outdegreeV[1] * IndegreeU[0]) / (CounterVU * CounterVU)
        #     FRNN1 = (outdegreeV[1] * IndegreeU[1])/(CounterVU * CounterVU)
        #     IPP1 = (IndegreeV[0] * IndegreeU[0])/(CounterVU * CounterVU)
        #     IPN1 = (IndegreeV[0] * IndegreeU[1]) / (CounterVU * CounterVU)
        #     INP1 = (IndegreeV[1] * IndegreeU[0]) / (CounterVU * CounterVU)
        #     INN1 = (IndegreeV[1] * IndegreeU[1]) / (CounterVU * CounterVU)
        #     OPP1 = (outdegreeV[0] * outdegreeU[0])/(CounterVU * CounterVU)
        #     OPN1 = (outdegreeV[0] * outdegreeU[1]) / (CounterVU * CounterVU)
        #     ONP1 = (outdegreeV[1] * outdegreeU[0]) / (CounterVU * CounterVU)
        #     ONN1 = (outdegreeV[1] * outdegreeU[1]) / (CounterVU * CounterVU)
        TempF.append(FPP)
        TempF.append(FPN)
        TempF.append(FNP)
        TempF.append(FNN)
        TempF.append(FRPP)
        TempF.append(FRPN)
        TempF.append(FRNP)
        TempF.append(FRNN)
        TempF.append(IPP)
        TempF.append(IPN)
        TempF.append(INP)
        TempF.append(INN)
        TempF.append(OPP)
        TempF.append(OPN)
        TempF.append(ONP)
        TempF.append(ONN)


        # TempF.append(FPP1)
        # TempF.append(FPN1)
        # TempF.append(FNP1)
        # TempF.append(FNN1)
        # TempF.append(FRPP1)
        # TempF.append(FRPN1)
        # TempF.append(FRNP1)
        # TempF.append(FRNN1)
        # TempF.append(IPP1)
        # TempF.append(IPN1)
        # TempF.append(INP1)
        # TempF.append(INN1)
        # TempF.append(OPP1)
        # TempF.append(OPN1)
        # TempF.append(ONP1)
        # TempF.append(ONN1)

        Feature_train[ii] = TempF
        labels_train[ii]=y_train[ii]
        G.add_path([u1, v1])
        Adjancy[u1][v1] = SignUV
        Adjancy[v1][u1] = SignVU
    end4 = datetime.datetime.now()
    print("F4", end4 - start4)
    print("#################################")
    print("Feature Train:", Feature_train.shape)
    print(" Size of training and testing")
    # lr = LogisticRegression()
    # lr.fit(Feature_train, labels_train.ravel())
    # clf = SVC()
    # clf = Ridge(alpha=1.0)
    # clf = linear_model.Lasso(alpha=0.1)
    clf = linear_model.SGDClassifier()
    clf.fit(Feature_train, labels_train.ravel())
    #####################################################
        ###Creat whole network
    for b in xrange(0,len(x_test)):
        nodeA = x_test[b][0]
        nodeB = x_test[b][1]
        Adjancy[nodeA][nodeB] += float(y_test[b])
        #G.add_path([nodeA, nodeB])
        G.add_edge(nodeA, nodeB)
    ######################################################
    print("Create Feature Matrix for Test Data")
    ##Feauter Matrix for test
    Feature_test=np.zeros((len(x_test), 16))
    labels_test=np.zeros((len(x_test), 1))
    for t in xrange(0, len(x_test)):
        Test = []
        CounterAB = 0
        CounterBA = 0
        A = x_test[t][0]
        B = x_test[t][1]
        degreeA = G.degree(A)
        degreeB = G.degree(B)
    #####################################################

        SignAB=Adjancy[A][B]
        SignBA=Adjancy[B][A]
        Adjancy[A][B]=0
        Adjancy[B][A]=0

        G.remove_edge(A,B)
        try:
            CounterAB = nx.shortest_path_length(G,A,B)
        except nx.exception.NetworkXNoPath:
            CounterAB = 0

        try:
            CounterBA = nx.shortest_path_length(G,B,A)
        except nx.exception.NetworkXNoPath:
            CounterBA = 0

#######################################################
        x = nx.neighbors(G, A)
        y = nx.neighbors(G, B)

        OutdegreeTA=OutDegree(x,Adjancy,A)
        sumOutA = OutdegreeTA[0] + OutdegreeTA[1]
        IndegreeTA = InDegree(x, Adjancy, A)
        sumInA = IndegreeTA[0] + IndegreeTA[1]
        IndegreeTB=InDegree(y,Adjancy,B)
        sumInB = IndegreeTB[0] + IndegreeTB[1]
        OutdegreeTB = OutDegree(y, Adjancy, B)
        sumOutB = OutdegreeTB[0] + OutdegreeTB[1]

        # Test.append(sumOutA)
        # Test.append(sumInA)
        # Test.append(sumInB)
        # Test.append(sumOutB)
        ###########################
        SucInA = successIn (x, Adjancy, A)
        SucOutA = successout(x, Adjancy, A)
        successA = SucInA[0] + SucOutA[0]
        # Test.append(successA)
        ##########################
        FailInA = SucInA[1] + SucOutA[1]
        # Test.append(FailInA)
        ##########################
        SucInB = successIn (x, Adjancy, B)
        SucOutB = successout(x, Adjancy, B)
        successB = SucInB[0] + SucOutB[0]
        # Test.append(successB)
        ##########################
        FailInB = SucInB[1] + SucOutB[1]
        # Test.append(FailInB)
        ##########################
        ##define new measure
        FPP = 0
        FPN = 0
        FNP = 0
        FNN = 0
        FRPP = 0
        FRPN = 0
        FRNP = 0
        FRNN = 0
        TIPP = 0
        TIPN = 0
        TINN = 0
        TINP = 0
        TOPP = 0
        TOPN = 0
        TONP = 0
        TONN = 0


        # FPP1 = 0
        # FPN1 = 0
        # FNP1 = 0
        # FNN1 = 0
        # FRPP1 = 0
        # FRPN1 = 0
        # FRNP1 = 0
        # FRNN1 = 0
        # TIPP1 = 0
        # TIPN1 = 0
        # TINN1 = 0
        # TINP1 = 0
        # TOPP1 = 0
        # TOPN1 = 0
        # TONP1 = 0
        # TONN1 = 0
        if CounterAB != 0:
            FPP = (OutdegreeTA[0] * IndegreeTB[0])/(CounterAB * CounterAB)
            FPN = (OutdegreeTA[0] * IndegreeTB[1])/(CounterAB * CounterAB)
            FNP = (OutdegreeTA[1] * IndegreeTB[0])/(CounterAB * CounterAB)
            FNN = (OutdegreeTA[1] * IndegreeTB[1])/(CounterAB * CounterAB)
            FRPP = (OutdegreeTB[0] * IndegreeTA[0])/(CounterAB * CounterAB)
            FRPN = (OutdegreeTB[0] * IndegreeTA[1])/(CounterAB * CounterAB)
            FRNP = (OutdegreeTB[1] * IndegreeTA[0]) / (CounterAB * CounterAB)
            FRNN = (OutdegreeTB[1] * IndegreeTA[1])/(CounterAB * CounterAB)
            TIPP = (IndegreeTB[0] * IndegreeTA[0]) / (CounterAB * CounterAB)
            TIPN = (IndegreeTB[0] * IndegreeTA[1]) / (CounterAB * CounterAB)
            TINP = (IndegreeTB[1] * IndegreeTA[0]) / (CounterAB * CounterAB)
            TINN = (IndegreeTB[1] * IndegreeTA[1]) / (CounterAB * CounterAB)
            TOPP = (OutdegreeTB[0] * OutdegreeTA[0]) / (CounterAB * CounterAB)
            TOPN = (OutdegreeTB[0] * OutdegreeTA[1]) / (CounterAB * CounterAB)
            TONP = (OutdegreeTB[1] * OutdegreeTA[0]) / (CounterAB * CounterAB)
            TONN = (OutdegreeTB[1] * OutdegreeTA[1]) / (CounterAB * CounterAB)



        # if CounterBA != 0:
        #     FPP1 = (OutdegreeTA[0] * IndegreeTB[0])/(CounterBA * CounterBA)
        #     FPN1 = (OutdegreeTA[0] * IndegreeTB[1])/(CounterBA * CounterBA)
        #     FNP1 = (OutdegreeTA[1] * IndegreeTB[0])/(CounterBA * CounterBA)
        #     FNN1 = (OutdegreeTA[1] * IndegreeTB[1])/(CounterBA * CounterBA)
        #     FRPP1 = (OutdegreeTB[0] * IndegreeTA[0])/(CounterBA * CounterBA)
        #     FRPN1 = (OutdegreeTB[0] * IndegreeTA[1])/(CounterBA * CounterBA)
        #     FRNP1 = (OutdegreeTB[1] * IndegreeTA[0]) / (CounterBA * CounterBA)
        #     FRNN1 = (OutdegreeTB[1] * IndegreeTA[1])/(CounterBA * CounterBA)
        #     TIPP1 = (IndegreeTB[0] * IndegreeTA[0]) / (CounterBA * CounterBA)
        #     TIPN1 = (IndegreeTB[0] * IndegreeTA[1]) / (CounterBA * CounterBA)
        #     TINP1 = (IndegreeTB[1] * IndegreeTA[0]) / (CounterBA * CounterBA)
        #     TINN1 = (IndegreeTB[1] * IndegreeTA[1]) / (CounterBA * CounterBA)
        #     TOPP1 = (OutdegreeTB[0] * OutdegreeTA[0]) / (CounterBA * CounterBA)
        #     TOPN1 = (OutdegreeTB[0] * OutdegreeTA[1]) / (CounterBA * CounterBA)
        #     TONP1 = (OutdegreeTB[1] * OutdegreeTA[0]) / (CounterBA * CounterBA)
        #     TONN1 = (OutdegreeTB[1] * OutdegreeTA[1]) / (CounterBA * CounterBA)
        Test.append(FPP)
        Test.append(FPN)
        Test.append(FNP)
        Test.append(FNN)
        Test.append(FRPP)
        Test.append(FRPN)
        Test.append(FRNP)
        Test.append(FRNN)
        Test.append(TIPP)
        Test.append(TIPN)
        Test.append(TINP)
        Test.append(TINN)
        Test.append(TOPP)
        Test.append(TOPN)
        Test.append(TONP)
        Test.append(TONN)

        # Test.append(FPP1)
        # Test.append(FPN1)
        # Test.append(FNP1)
        # Test.append(FNN1)
        # Test.append(FRPP1)
        # Test.append(FRPN1)
        # Test.append(FRNP1)
        # Test.append(FRNN1)
        # Test.append(TIPP1)
        # Test.append(TIPN1)
        # Test.append(TINP1)
        # Test.append(TINN1)
        # Test.append(TOPP1)
        # Test.append(TOPN1)
        # Test.append(TONP1)
        # Test.append(TONN1)
        Feature_test[t] = Test
        G.add_path([A, B])
        labels_test[t] = y_test[t]
        Adjancy[A][B] = SignAB
        Adjancy[B][A] = SignBA
################################################
    print("Predict Process")
    # predict=lr.predict(Feature_test)
    predict=clf.predict(Feature_test)
    print("Confusion matrix", confusion_matrix(labels_test, predict))
    Accuracy = accuracy_score(labels_test, predict)
    main_f+=Accuracy
    print("Accuracy Score:", Accuracy)
    ######################################
    Adjancy = None
    print(classification_report(labels_test, predict, target_names=target_names))
print("########################################")
print("f avarage", main_f/n_folds)

