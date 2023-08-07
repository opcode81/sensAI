import re

import torch
from sklearn.preprocessing import StandardScaler

import sensai.torch
from sensai import NormalisationMode, normalisation
from sensai.data_transformation import DFTNormalisation, DFTSkLearnTransformer
from sensai.featuregen import FeatureGeneratorTakeColumns
from sensai.torch import NNOptimiser, Optimiser
from sensai.torch.torch_base import TorchModelFromModuleFactory
from sensai.torch.torch_data import TorchDataSetFromTensors
from sensai.torch.torch_models.mlp.mlp_models import MultiLayerPerceptronVectorRegressionModel
from sensai.torch.torch_models.residualffn.residualffn_models import ResidualFeedForwardNetworkVectorRegressionModel
from sensai.torch.torch_modules import MultiLayerPerceptron
from sensai.torch.torch_opt import NNLossEvaluatorClassification, NNOptimiserParams


def test_classifier_MLPClassifier(irisClassificationTestCase):
    featureNames = irisClassificationTestCase.data.inputs.columns
    dftNorm = DFTNormalisation([DFTNormalisation.Rule(re.escape(f)) for f in featureNames],
        default_transformer_factory=StandardScaler)
    nn_optimiser_params = NNOptimiserParams(epochs=100, optimiser="adam", batch_size=200)
    model = sensai.torch.models.MultiLayerPerceptronVectorClassificationModel(
            hidden_dims=(50,25,8), cuda=False,
            normalisation_mode=NormalisationMode.NONE, hid_activation_function=torch.tanh,
            nn_optimiser_params=nn_optimiser_params) \
        .with_name("torchMLPClassifier") \
        .with_input_transformers([dftNorm]) \
        .with_feature_generator(FeatureGeneratorTakeColumns())
    irisClassificationTestCase.testMinAccuracy(model, 0.8)


def test_NNOptimiserWithoutValidation_MLPClassifier(irisDataSet):
    """
    Tests and demonstrates the use of NNOptimiser without validation on the iris data set, using low-level
    interfaces (i.e. without high-level abstractions such as VectorModel)
    """
    iodata = irisDataSet.getInputOutputData()
    outputs = iodata.outputs.iloc[:, 0]
    classLabels = list(outputs.unique())
    outputTensor = torch.tensor([classLabels.index(l) for l in outputs])
    scaler = normalisation.VectorDataScaler(iodata.inputs, normalisation.NormalisationMode.MAX_BY_COLUMN)
    inputTensor = torch.tensor(scaler.get_normalised_array(iodata.inputs), dtype=torch.float32)
    dataSet = TorchDataSetFromTensors(inputTensor, outputTensor, False)
    model = TorchModelFromModuleFactory(lambda: MultiLayerPerceptron(inputTensor.shape[1], len(classLabels),
            (4, 3), hid_activation_fn=torch.tanh, output_activation_fn=None), cuda=False)
    NNOptimiser(NNOptimiserParams(loss_evaluator=NNLossEvaluatorClassification(NNLossEvaluatorClassification.LossFunction.CROSSENTROPY), train_fraction=1.0, epochs=300,
            optimiser="adam")).fit(model, dataSet)
    modelOutputs = model.apply(inputTensor, as_numpy=False)
    accuracy = torch.sum(torch.argmax(modelOutputs, 1) == outputTensor).item() / len(outputTensor)
    assert accuracy > 0.9


def test_regressor_ResidualFeedForwardNetworkVectorRegressionModel(diabetesRegressionTestCase):
    nnOptimiserParams = NNOptimiserParams(optimiser=Optimiser.ADAMW, batch_size=32, early_stopping_epochs=30)
    model = ResidualFeedForwardNetworkVectorRegressionModel((20, 20, 20), bottleneck_dimension_factor=0.5, cuda=False, nn_optimiser_params=nnOptimiserParams) \
        .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
        .with_name("RFFN")
    diabetesRegressionTestCase.testMinR2(model, 0.48)


def test_regressor_MLP(diabetesRegressionTestCase):
    nnOptimiserParams = NNOptimiserParams(optimiser=Optimiser.ADAMW, batch_size=32, early_stopping_epochs=30)
    model = MultiLayerPerceptronVectorRegressionModel((20, 20), cuda=False, nn_optimiser_params=nnOptimiserParams, hid_activation_function=torch.relu) \
        .with_feature_transformers(DFTSkLearnTransformer(StandardScaler())) \
        .with_name("MLP")
    diabetesRegressionTestCase.testMinR2(model, 0.48)
