tensorflow models:
-	All trained models and histories are saved in the models directory for the datasets cifar and monkeys (both are dictionaries in the model dictionary)
- 	Each model starting with tf (tensorflow) can be started by opening the file. The nomenclature is tf_model-architecture_dataset.py.
	The script checks, if a trained model is available or will start training and predicting otherwise. Trained models can be downloaded from https://github.com/nadiaverdha/MachineLearningEx3/tree/main (copy the model folder to your cwd)
- 	Generally, the data must be in a "data" folder in the current working directory (os.getcwd()) following the initial download structure.
 	import_cifar.py requires the described folder structure from the download from http://www.cs.toronto.edu/~kriz/cifar.html
-	evaluate_nn_models starts an evaluation process of all models based on tensorflow.keras. If the models are not train it will result in an error.
	The provided run_model() function allows the display and save of train history and convolutionmatrix as well as a print of the model architecture.
