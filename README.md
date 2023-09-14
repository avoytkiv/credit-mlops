# Customers segmentation: MLOps approach

## Problem

Understand clients’ behaviour having more than hundred of features and millions of records. Why? For targeted marketing. 

<img width="834" alt="Screenshot 2023-09-13 at 21 48 08" src="https://github.com/avoytkiv/credit-mlops/assets/74664634/e38b76b4-2feb-4242-a9c7-1a4f5ea733da">

Why MLOps?

Challenges with ML projects:

**Data Management**: data changes frequently, it impacts analysis, managing versions of datasets is a pain.
**Experiments management**: different versions of data, code and parameteres make it impossible to track experiments in a consistent and reliable way.
**Deployment management**: different environments might require different dependencies and configurations.


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

With the best experiment in mind, we: 
- trigger a workflow from the command line (CLI) or DVC Studio on experiments branch by pushing changes to remote 
- which in turn will invoke a workflow job (with the help of GitHub Actions), in our case two jobs:
  - deploy cloud runner (AWS EC2 instance),install all dependencies
  - run training 
- that will result in a report pushed back into Git and published as PR (Pull Request) comment including metrics, plots. 
- After reviewing and approving this PR, the new model will be in the production branch

<img width="1154" alt="Screenshot 2023-09-13 at 21 07 57" src="https://github.com/avoytkiv/credit-mlops/assets/74664634/08168b7f-75d4-4ce5-a8b4-412eb948bcc7">


### Continuous Deployment

From here, we can trigger another workflow by pushing a predefined tag. Finally, it will deploy our model to the `Heroku` using `FastAPI Endpoint`. So anyone with access can use it for prediction.

## Interpreting results

Compare selected features across our clusters. Though the plot below is unreadable, its purpose is to give a general idea of how to start thinking about the problem. 

![comparison](https://github.com/avoytkiv/credit-mlops/assets/74664634/69eef1b7-7d09-48d3-977f-c985eaae5c7a)

Summarizing data grouped by clusters across selected features.

<img width="1139" alt="Customer_Segmentation_by_clustering_-_Google_Slides" src="https://github.com/avoytkiv/credit-mlops/assets/74664634/1d631b39-0ed6-4ecb-8b20-b79ed4bf9094">   


From here start consulting with the business team. 

## Environment Variables

To run this project, you will need to add the following environment variables to your repo secrets (Go to repo --> Settings --> Secrets and Variables --> Actions):
`AWS_ACCESS_KEY_ID` - to run experiments on more powerful cloud instance; 
`AWS_SECRET_ACCESS_KEY` - see above;
`HEROKU_API_KEY` - to deploy app;
`TOKEN_GITHUB` - to give permission for Github workflows.


## TODO

- [ ] Autoencoder: Debug negative training loss.
- [ ] Create user interface for the app with Streamlit.
- [ ] Documentation.
- [ ] Packaging.
- [ ] Log metrics with DVC Live.
- [ ] Validate data and model.
- [ ] Implement testing (pytest).
- [ ] Integrate with DagsHub - same as Git but can see not only hashed data but data itself.
