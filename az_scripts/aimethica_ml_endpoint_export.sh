#!/bin/bash

# Variables
RESOURCE_GROUP="AiMethica-ML"
WORKSPACE_NAME="AiMethica-AZML"
OUTPUT_DIR="./endpoints"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Get the list of endpoints
endpoints=$(az ml online-endpoint list --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --query "[].name" -o tsv)

# Loop through each endpoint and export its details and deployments
for endpoint in $endpoints; do
    echo "Exporting endpoint: $endpoint"
    
    # Export endpoint details
    az ml online-endpoint show --name $endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME -o yaml > "$OUTPUT_DIR/${endpoint}_endpoint.yaml"
    
    # Get the list of deployments for the endpoint
    deployments=$(az ml online-deployment list --endpoint-name $endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --query "[].name" -o tsv)
    
    # Loop through each deployment and export its details
    for deployment in $deployments; do
        echo "Exporting deployment: $deployment for endpoint: $endpoint"
        az ml online-deployment show --name $deployment --endpoint-name $endpoint --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME -o yaml > "$OUTPUT_DIR/${endpoint}_${deployment}_deployment.yaml"
    done
done

echo "Export completed."