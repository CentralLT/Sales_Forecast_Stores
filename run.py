import make_dataset
import build_features
import build_model
import d6tflow
import cfg

""" Delete # to uncomment commands and run script, d6tflow will automatically run all necessary tasks and save
output in ../../Data/output/ """

# Create data set

#d6tflow.run(make_dataset.TaskMakeDataSet())

# Create Features

#d6tflow.run(build_features.TaskRollingWindow())
#d6tflow.run(build_features.TaskLagStructure())
#d6tflow.run(build_features.TaskMeanEncoding())
#d6tflow.run(build_features.TaskDownCast())
#d6tflow.run(build_features.TaskLogChange(), forced_all=True)
#d6tflow.run(build_features.TaskTrainTestSplit(), forced_all=True)

# Run Model

# Baseline Model

#d6tflow.run(build_model.TaskBuildBaseline(), forced_all=True)

# Linear Models

#d6tflow.run(build_model.TaskLinearModel(), forced_all=True)
#d6tflow.run(build_model.TaskLinearModelTimeHorizon(), forced_all=True)
#d6tflow.run(build_model.TaskRidgeModel(), forced_all=True)
#d6tflow.run(build_model.TaskLinearModelSelect(), forced_all=True)

# Random Forest

#d6tflow.run(build_model.TaskRandomForest(), forced_all=True)

# kNN
#d6tflow.run(build_model.TaskKNN(), forced_all=True)

# Neural Network

# Make Data Set for NN

d6tflow.run(build_model.TaskDNN(), forced_all=True)
