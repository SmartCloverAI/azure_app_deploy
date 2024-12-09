docker build -t aidamian/azureml-dev-slim -f Dockerfile_azureml .
docker push aidamian/azureml-dev-slim


docker build -t aidamian/azureml-dev-tf -f Dockerfile_tf .
docker push aidamian/azureml-dev-tf