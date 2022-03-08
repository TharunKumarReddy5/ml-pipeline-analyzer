import evalml
from evalml.automl import AutoMLSearch


class PipelineSuggest:
    """
    PiplineSuggest class that suggests the best ML pipeline for a given dataset
    """
    def __init__(self):
        # Add support for placeholder
        self.data = None
        self.response = None
        self.predictor_list = None
        self.problem_type = None
        self.objective = None
        self.test_size = None
        self.best_pipeline = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        pass

    def fit(self, data, response, predictor_list, problem_type, objective='auto', test_size=0.2):
        """
        Initializes the attributes for the PipelineSuggest class and divides the data into train and test split

        Args:
            data (pd.DataFrame): The input data to find the best ML Pipeline. Required.
            response (str): The response/target variable to be used for training. Required.
            predictor_list (list): A list of predictor variables (str) to be used for training. Required.
            problem_type (str): Types of supervised learning problem. Required.
            objective (str): The objective to optimize for to find the best pipeline. Required. Defaults to 'auto'.
                When set to 'auto', chooses:
                    - LogLossBinary for binary classification problems,
                    - LogLossMulticlass for multiclass classification problems, and
                    - R2 for regression problems.
            test_size (float): Percentage of test data after train test split. Optional. Defaults to 0.2

        Returns:
            None
        """
        self.data = data
        self.response = data[response]
        self.predictor_list = data[predictor_list]
        self.problem_type = problem_type
        self.objective = objective
        self.test_size = test_size
        self.best_pipeline = None

        y = data[response]
        x = data[predictor_list]
        self.x_train, self.x_test, self.y_train, self.y_test = evalml.preprocessing.split_data(x, y, problem_type,
                                                                                               test_size)

    def _suggest_helper(self):
        automl = AutoMLSearch(X_train=self.x_train, y_train=self.y_train, problem_type=self.problem_type,
                              objective=self.objective)
        automl.search()
        self.best_pipeline = automl.best_pipeline
        self.best_model = self.best_pipeline.estimator
        # TODO: Need to add attribute for best_fe
        # self.best_fe = Pipeline for the fe

    def suggest(self, suggest_type='all'):
        """
        Finds the best ML pipeline/model/feature engineering steps

        Args:
            suggest_type (str): The best pipeline part to suggest. Optional. Defaults to 'all'.
                'fe' - Suggests the feature engineering steps of the best ML pipeline.
                'model' - Suggests the model of the best ML pipeline.
                'all' - Suggests the best ML pipeline.
        Returns:
            obj (evalml): Returns evalml object specific to the problem type. This object can be used to access
                        additional evalml attributes and methods
        """
        if self.best_pipeline is None:
            self._suggest_helper()

        best_pipeline_summary = self.best_pipeline.summary
        best_pipeline_components = best_pipeline_summary.split('w/')

        if suggest_type == 'fe':
            return best_pipeline_components[1].strip()

        elif suggest_type == 'model':
            return best_pipeline_components[0].strip()

        return self.best_pipeline
