import pandas as pd
from mlpipeline_analyzer.suggest import PipelineSuggest


class TestPipelineSuggest:
    def test_fit(self):
        """
        Test that the data is divided into train and test data before finding the best ML pipeline
        """

        data = pd.read_csv('sample_data/income_classification.csv')
        response = 'income'
        predictor = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country']

        problem_type = 'binary'
        objective = 'F1'

        ps = PipelineSuggest()
        ps.fit(data, response, predictor, problem_type, objective)

        assert (ps.x_train.shape[0] != 0) is True
        assert (ps.x_test.shape[0] != 0) is True
        assert (ps.y_train.shape[0] != 0) is True
        assert (ps.y_test.shape[0] != 0) is True

