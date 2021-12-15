
class GBDT():

    def __init__(self, max_tree_num=3):
        """
        In this function you need to initialize the Gradient Boosting Decision Tree
        :param  max_tree_num: Maximum number of boosting tree,default = 3
        """
        # -- write your code here --
        self.tree_list = []
        self.max_tree_num = max_tree_num

    def fit(self, x, y):
        """
        In this function you need to fit the input data and label to the decision Tree
        :param x: Training set X
        :param y: Training set Y
        """
        # -- write your code here --
        residual = y
        for i in range(self.max_tree_num):
            model = DecisionTreeClassifier()
            model.fit(x, residual)
            self.tree_list.append(model)
            prediction = model.predict(x)
            residual = residual - prediction

    def predict(self, x):
        """
        In this function you need to use the classifier to predict input test data
        :param x: Input test data which format is ndarray
        :return: Return the prediction you got from model
        """
        # -- write your code here --
        # x.shape = (data_points, feature)
        y = np.zeros(x.shape[0])
        for model in self.tree_list:
            new_pred = np.array(model.predict(x))
            y += new_pred
        return y