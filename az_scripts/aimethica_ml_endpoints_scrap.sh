az ml online-endpoint show --name weekly-company-cashflow  --resource-group AiMethica-ML --workspace-name AiMethica-AZML > weekly-company-cashflow_managed.json
az ml online-deployment list --endpoint-name weekly-company-cashflow --resource-group AiMethica-ML --workspace-name AiMethica-AZML
az ml online-deployment show --name tenderrootwdxkv267-1 --endpoint-name weekly-company-cashflow --resource-group AiMethica-ML --workspace-name AiMethica-AZML
az ml environment show --name aimethica-anomaly-env --version 7 --resource-group AiMethica-ML --workspace-name AiMethica-AZML -o yaml > aimethica-anomaly-env.yaml

az ml online-deployment create -f tenderrootwdxkv267-1.yaml --resource-group AiMethica-ML --workspace-name AiMethica-AZML
az ml online-deployment get-logs --endpoint-name anomaly-document-nir --name tff-blue --lines 100 --resource-group AiMethica-ML --workspace-name AiMethica-AZML

az ml workspace show  --resource-group AiMethica-ML --name AiMethica-AZML

