# Project Title

A brief description of what this project does and who it's for

## ML Pipeline Architecture

The pipeline takes code, parameters, and data as inputs and moves through various stages to generate outputs, including metrics, models, and plots.
Create a __reproducible__ machine learning pipeline by defining stages in `dvc.yaml` file. 
My pipeline consists of four distinct stages: __preprocessing, featurization, training, and visualization__. A unique Python script corresponds to each of these stages, serving as the command to execute that particular phase. All stages rely on parameters that are defined in the `params.yaml` file. Additionally, each stage produces outputs, which may include intermediate data, the model itself, relevant metrics, or plots.

<img width="1607" alt="pipeline graph" src="https://github.com/avoytkiv/credit-mlops/assets/74664634/eca6b531-269d-4d03-a946-fd0b3fa4b431">

## Demo

### Running ML experiments

Running a single experiment is just one command with on-the-fly updates to parameters without needing to alter code or configuration files

> Thanks to DVC's smart caching mechanism, our pipeline can detect stages that haven't been altered since the last run. By recognizing and skipping these unchanged stages, we conserved both time and compute resources, enhancing efficiency.

```shell
$ dvc exp run \
	-S 'featurize=autoencoder' \
	-S 'train/model=kmeans' 
	
Reproducing experiment 'quiet-juba'                                                                                                                                                                     
Stage 'preprocess' didn't change, skipping                                                                                                                                                              
Running stage 'featurize':
...
```

#### Running multiple experiments

We can also load multiple config groups in an experiments queue for example to run a grid search of ML hyperparameters. 

```shell
$ dvc exp run --queue \
	-S 'featurize=autoencoder' \
	-S 'train/model=hdbscan,kmeans' \
	-S 'featurize.parameters.encoding_dim=6,8,10' \
	-S 'featurize.hyperparameters.batch_size=16,64,128' \
	-S 'featurize.hyperparameters.loss=adam,sgd' \
	-S 'featurize.hyperparameters.loss=mean_squared_error,binary_crossentropy' 
	
$ dvc queue start
```

The output of this command is a table that lists all the experiments (DVC Plugin in VS Code).

#### Track experiments

<img width="1741" alt="experiments1" src="https://github.com/avoytkiv/credit-mlops/assets/74664634/8553ea39-ddca-4f7b-83e7-68a8dcb45aed">

<img width="1741" alt="experiments2" src="https://github.com/avoytkiv/credit-mlops/assets/74664634/b58d976c-be61-4eae-8bad-97d6dbd1d0c4">

### Continuous Integration

### Continuous Deployment

## Environment Variables

To run this project, you will need to add the following environment variables to your repo secrets (Go to repo --> Settings --> Secrets and Variables --> Actions):
`AWS_ACCESS_KEY_ID` - to run experiments on more powerful cloud instance; 
`AWS_SECRET_ACCESS_KEY` - see above;
`HEROKU_API_KEY` - to deploy app;
`TOKEN_GITHUB` - to give permissions for Github workflows.


## TODO

- [ ] Autoencoder: Debug negative training loss.
- [ ] Create user interface for the app with Streamlit.
- [ ] Documentation.
- [ ] Packaging.
- [ ] Log metrics with DVC Live.
- [ ] Validate data and model.
- [ ] Implement testing (pytest).
- [ ] Integrate with DagsHub - same as Git but can see not only hashed data but data itself.
