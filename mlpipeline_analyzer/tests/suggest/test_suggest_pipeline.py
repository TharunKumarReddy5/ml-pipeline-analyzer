import unittest
import pandas as pd
from mlpipeline_analyzer.suggest import PipelineSuggestion


class TestSuggest(unittest.TestCase):

    def test_fit(self):
        """
        Test that the data is divided into train and test data before finding the best ML pipeline
        """

        data = pd.read_csv('./mlpipeline_analyzer/tests/suggest/sample_data/income_classification.csv')
        response = 'income'
        predictor = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country']

        problem_type = 'binary'
        objective = 'F1'

        ps = PipelineSuggestion.PipelineSuggest()
        ps.fit(data, response, predictor, problem_type, objective)

        self.assertTrue(ps.x_train.shape[0])
        self.assertTrue(ps.x_test.shape[0])
        self.assertTrue(ps.y_train.shape[0])
        self.assertTrue(ps.y_test.shape[0])


if __name__ == '__main__':
    unittest.main()
