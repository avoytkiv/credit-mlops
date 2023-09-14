# Project Title

A brief description of what this project does and who it's for

## ML Pipeline Architecture

## Demo

### Running ML experiments

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
