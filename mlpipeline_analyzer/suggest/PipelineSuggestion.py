import evalml
from evalml.automl import AutoMLSearch

class PipelineSuggest():
    def __init__(self):
        # Add support for placeholder
        pass
    
    def fit(self, data, response, predictor_list, problem_type, objective = 'auto', test_size = 0.2):
        '''
        This function takes the inputs from the user and parses the data to initialize the attributes and create the x_train, x_test, y_train, y_test objects.
            - The user provides with a dataframe as an input for data.
            - response is the response variable or the dependent variable.
            - predictor_list is the list of the predictors/independent variables.
            - problem_type would be one of the supported problem_types by the evalml library.
            - test_size is set to default 20%.
        '''
        self.data = data
        self.response = data[response]
        self.predictor_list = data[predictor_list]
        self.problem_type = problem_type
        self.objective = objective
        self.test_size = test_size
        self.best_pipeline = None

        y = data[response]
        X = data[predictor_list]
        self.x_train, self.x_test, self.y_train, self.y_test = evalml.preprocessing.split_data(X, y, problem_type, test_size)
        
    def _suggest_helper(self):
        automl = AutoMLSearch(X_train=self.x_train, y_train=self.y_train, problem_type=self.problem_type, objective=self.objective)
        automl.search()
        self.best_pipeline = automl.best_pipeline
        self.best_model = self.best_pipeline.estimator 
        # TODO: Need to add attribute for best_fe
        # self.best_fe = Pipeline for the fe
    
    def suggest(self, suggest_type = 'all'):
        if self.best_pipeline is None:
            self._suggest_helper()
        
        best_pipeline_summary = self.best_pipeline.summary    
        best_pipeline_components = best_pipeline_summary.split('w/')

        if suggest_type == 'fe':
            return best_pipeline_components[1].strip()
            
        elif suggest_type == 'model':
            return best_pipeline_components[0].strip()
        
        return self.best_pipeline
