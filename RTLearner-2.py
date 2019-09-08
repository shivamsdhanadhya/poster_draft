import numpy as np
from random import randint

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size=leaf_size
        self.random_tree=None

    def addEvidence(self,dataX,dataY):
        """
            Method to train RT Learner
        """
        dataY = np.array([dataY]).T
        data = np.append(dataX, dataY, axis=1)
        self.random_tree = self.build_tree(data)

    def query(self, Xtest):
        """
            Function to query the model based on the input Xtest values
        """
        query_res = []; row_cnt = Xtest.shape[0]
        for i in range(row_cnt):
            value = self.query_random_tree(Xtest[i, :])
            query_res.append(float(value))
        return query_res


    def build_tree(self, data):
        """
            Method to build RT
        """
        #Base condition #1
        if data.shape[0] <= self.leaf_size:
            return np.array([["leaf", np.mean(data[:, -1]), np.nan, np.nan]])
        #Base condition #2
        if np.all(data[0, -1] == data[:, -1], axis=0):
            return np.array([["leaf", data[0, -1], np.nan, np.nan]])
        else:
            #Get the random feature
            random_feature = randint(0,data.shape[1]-2)
            #Get the split value using median calculations
            split_value = np.median(data[:, random_feature])
            if max(data[:, random_feature]) == split_value:
                #Formation of left subtree only
                return np.array([["leaf", np.mean(data[:, -1]), np.nan, np.nan]])
            #Recursive call For Left Subtree formation
            left_subtree=self.build_tree(data[data[:, random_feature] <= split_value])
            #Recursive call For Right Subtree formation
            right_subtree=self.build_tree(data[data[:, random_feature] > split_value])
            root = np.array([[random_feature,split_value,1,left_subtree.shape[0]+1]])
            new_tree = np.append(root, left_subtree, axis=0)
            return np.append(new_tree, right_subtree, axis=0)

    def random_feature(self,data):
        """
            Function to get the random feature
        """
        max_corr_coef = 0; random_feature = 0
        #Get Xdata
        dataX = data.shape[1] - 1
        #Get Ydata
        dataY = data[:, data.shape[1] - 1]
        corr_coef_arr = []
        for i in range(dataX):
            corr_coef = np.corrcoef(data[:, i], dataY)
            corr_coef = abs(corr_coef[0, 1])
            corr_coef_arr.append(corr_coef)
        for i in range(len(corr_coef_arr)):
            if corr_coef_arr[i] > max_corr_coef:
                max_corr_coef = corr_coef_arr[i]
                random_feature = i
        return int(random_feature)

    def query_random_tree(self, value):
        """
            Function to query the random Tree internally
        """
        row_index = 0
        #if not a leaf node
        while(self.random_tree[row_index, 0] != "leaf"):
            feature = self.random_tree[row_index, 0]
            split_value = self.random_tree[row_index, 1]
            if value[int(float(feature))] <= float(split_value):
                #Move to Left
                row_index = row_index + int(float(self.random_tree[row_index, 2]))
            else:
                #Move to Right
                row_index  = row_index + int(float(self.random_tree[row_index, 3]))
        #Otherwise it is a leaf node
        return self.random_tree[row_index, 1]

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
