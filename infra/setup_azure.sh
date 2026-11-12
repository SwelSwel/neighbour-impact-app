#!/usr/bin/env bash
# One-time Azure set-up for the neighbour-impact app. Run once, from Azure Cloud Shell or any machine with the Azure CLI.
# After this, every push to main deploys automatically through GitHub Actions (.github/workflows/ci-cd.yml).
set -euo pipefail

# ---- edit these ----------------------------------------------------------------------------------------------
LOCATION="eastasia"                      # Hong Kong. Azure for Students only allows some regions: list yours with
                                         #   az policy assignment list --query "[].parameters" -o json
RG="rg-neighbour-impact"
ACR="acrneighbour$RANDOM"                # must be globally unique, lowercase letters/digits only
ACA_ENV="aca-neighbour-env"
LAW="law-neighbour-impact"
IDENTITY="id-neighbour-acrpull"
API_APP="neighbour-api"
UI_APP="neighbour-ui"
GITHUB_REPO="SwelSwel/neighbour-impact-app"   # owner/repo; the repo must exist before you run this
# ---------------------------------------------------------------------------------------------------------------

SUB_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
echo "Subscription: $SUB_ID"

echo "1. Register resource providers"
for p in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry Microsoft.ManagedIdentity; do
  az provider register --namespace "$p" --wait -o none
done
az extension add --name containerapp --upgrade -o none

echo "2. Resource group, container registry (admin user off), log workspace"
az group create -n "$RG" -l "$LOCATION" -o none
az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled false -o none
az monitor log-analytics workspace create -n "$LAW" -g "$RG" -l "$LOCATION" -o none
LAW_ID=$(az monitor log-analytics workspace show -n "$LAW" -g "$RG" --query customerId -o tsv)
LAW_KEY=$(az monitor log-analytics workspace get-shared-keys -n "$LAW" -g "$RG" --query primarySharedKey -o tsv)

echo "3. Container Apps environment (sends container logs to Log Analytics)"
az containerapp env create -n "$ACA_ENV" -g "$RG" -l "$LOCATION" \
  --logs-workspace-id "$LAW_ID" --logs-workspace-key "$LAW_KEY" -o none

echo "4. Managed identity that the apps use to pull images (no registry passwords)"
az identity create -n "$IDENTITY" -g "$RG" -o none
IDENTITY_ID=$(az identity show -n "$IDENTITY" -g "$RG" --query id -o tsv)
IDENTITY_PRINCIPAL=$(az identity show -n "$IDENTITY" -g "$RG" --query principalId -o tsv)
ACR_ID=$(az acr show -n "$ACR" --query id -o tsv)
az role assignment create --assignee-object-id "$IDENTITY_PRINCIPAL" --assignee-principal-type ServicePrincipal \
  --role AcrPull --scope "$ACR_ID" -o none

echo "5. GitHub Actions identity: OpenID Connect federated credential (no client secret stored anywhere)"
APP_ID=$(az ad app create --display-name "gh-neighbour-impact-deploy" --query appId -o tsv)
SP_ID=$(az ad sp create --id "$APP_ID" --query id -o tsv)
az ad app federated-credential create --id "$APP_ID" --parameters "{
  \"name\": \"github-main\", \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:$GITHUB_REPO:environment:production\", \"audiences\": [\"api://AzureADTokenExchange\"]}" -o none
# GitHub now sends a subject that also contains the owner's and repo's numeric IDs
# (repo:owner@ownerId/repo@repoId:environment:production), so trust that form too.
GH_IDS=$(curl -fsS "https://api.github.com/repos/$GITHUB_REPO" | jq -r '"\(.owner.login)@\(.owner.id)/\(.name)@\(.id)"' || true)
if [ -n "$GH_IDS" ]; then
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"github-production-ids\", \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"repo:$GH_IDS:environment:production\", \"audiences\": [\"api://AzureADTokenExchange\"]}" -o none
else
  echo "   Could not read the repo IDs from GitHub. If the login step fails, add the subject printed in its error message."
fi
RG_ID=$(az group show -n "$RG" --query id -o tsv)
az role assignment create --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
  --role Contributor --scope "$RG_ID" -o none
az role assignment create --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
  --role AcrPush --scope "$ACR_ID" -o none
# Lets the pipeline attach the pull identity to the apps it creates
az role assignment create --assignee-object-id "$SP_ID" --assignee-principal-type ServicePrincipal \
  --role "Managed Identity Operator" --scope "$IDENTITY_ID" -o none

cat <<EOF

Done. Now add these in GitHub -> repository Settings:

  Environments -> New environment "production"
  Secrets and variables -> Actions -> Secrets:
    AZURE_CLIENT_ID        $APP_ID
    AZURE_TENANT_ID        $TENANT_ID
    AZURE_SUBSCRIPTION_ID  $SUB_ID
    API_KEY                (choose a long random value, e.g. output of: openssl rand -hex 24)
  Secrets and variables -> Actions -> Variables:
    RESOURCE_GROUP         $RG
    ACR_NAME               $ACR
    ACA_ENV                $ACA_ENV
    API_APP                $API_APP
    UI_APP                 $UI_APP
    PULL_IDENTITY_ID       $IDENTITY_ID

Then push to main (or run the workflow manually) to build, test and deploy.
EOF
