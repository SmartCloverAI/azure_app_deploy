#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, ManagedIdentityCredential


# In[19]:


AZURE_DIRECTORY = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/aimethica-general/code/Users/alexandru.popescu/anomaly_DOCUMENTE_FACTURI_EXPORT'


# In[20]:


subscription_id = "9107822e-69ab-454f-a905-b59c4e15c1af"
resource_group = "AiMethica-ML"
workspace = "AiMethica-AZML"


# In[21]:


ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)


# In[22]:


ws = ml_client.workspaces.get(workspace)
print(ws.location, ":", ws.resource_group)


# In[23]:


custom_env_name = 'aimethica-anomaly-env'
job_name = 'careful_map_52rd1875dq'
env_name = custom_env_name + "@latest"


# In[24]:


from azure.ai.ml.entities import Model

model = Model(
    # the script stores the model as "model"
    path=f"azureml://jobs/{job_name}/outputs/artifacts/paths/outputs/",
    name="anomaly_document_facturi_export",
    description="Anomaly detection model for the DOCUMENT FACTURI EXPORT dataset.",
    type="custom_model"
)


# In[25]:


registered_model = ml_client.models.create_or_update(model=model)


# In[26]:


online_endpoint_name = "anomaly-document-facturi-export"


# In[27]:


# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Anomaly detection model for the DOCUMENT FACTURI EXPORT dataset.",
    auth_mode="key"
)

endpoint = ml_client.begin_create_or_update(endpoint).result()

print(f"Endpint {endpoint.name} provisioning state: {endpoint.provisioning_state}")


# In[28]:


model_name = registered_model.name + '@latest'

code_directory = os.path.join(AZURE_DIRECTORY,'src')

blue_deployment = ManagedOnlineDeployment(
    name="tff-blue",
    endpoint_name=online_endpoint_name,
    model=model_name,
    code_configuration=CodeConfiguration(code=code_directory, scoring_script="score.py"),
    environment=env_name,
    instance_type="Standard_DS1_v2",
    instance_count=1,
)


# In[29]:


blue_deployment = ml_client.begin_create_or_update(blue_deployment)


# In[30]:


anomaly_request = os.path.join(AZURE_DIRECTORY,"data/anomaly_request.json")
ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="tff-blue",
    request_file=anomaly_request
)

