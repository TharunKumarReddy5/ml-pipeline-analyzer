from mlpipeline_analyzer import PipelineDiagram
from sklearn.svm import SVC
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.discriminant_analysis import *
from sklearn.impute import *
import numpy as np

def sum():
    return
    
model = SVC(C=1.0, kernel='poly', degree=5, gamma='scale')
pipeline = Pipeline([('custom',sum()),('labelencoder',LabelEncoder()), #-- Pipe Transformer 1
                     ("imputer", SimpleImputer(missing_values=np.nan, strategy='mean')), #-- Pipe Transformer 2
                     ('scale', FeatureUnion([
                ('minmax', MinMaxScaler()),    #-- Parallel Transformer 3
                ('standardscaler', StandardScaler()),  #-- Parallel Transformer 4
                ('normalize', Normalizer())])),#-- Parallel Transformer 5
            ("LDA", LinearDiscriminantAnalysis()), #-- Pipe Transformer 6
	     ('classifier', model)]) 	      #-- Pipe Classifier/Predictor 7

