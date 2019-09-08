import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose = False):
        self.learner = learner
        self.bags = bags
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self, dataX, dataY):
        """
            Method to train Bag Learner
        """
        indices = np.linspace(0, dataX.shape[0]-1, dataX.shape[0]).astype(int)
        for learner in self.learners:
            #Make some random choice
            index = np.random.choice(indices, indices.size)
            #Make the learner specific model train
            learner.addEvidence(dataX.take(index, axis=0), dataY.take(index, axis=0))

    def query(self, Xtest):
        """
            Function to query the model based on the input Xtest values
        """
        query_arr = []
        for learner in self.learners:
            query_arr.append(learner.query(Xtest))
        #Get the query result
        query_res = np.array(query_arr)
        return np.mean(query_res, axis=0).tolist()

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
