LOCATION="your_region"
REPOSITORY_NAME="llm-docker-repo"
IMAGE_NAME="llama_cpp_api_image"
PROJECT_ID="your_project_id"
MODEL_DISPLAY_NAME="llama_cpp_model"
ENDPOINT_DISPLAY_NAME="llama_cpp_model_endpoint"
DEPLOYMENT_NAME="llama_cpp_model_deployment"

# build with cloudbuild
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_LOCATION=$LOCATION,_REPOSITORY=$REPOSITORY_NAME,_IMAGE=$IMAGE_NAME .
  
# upload model
# make sure the port is same as the port set for docker container
gcloud ai models upload \
  --region=europe-west3 \
  --display-name=$MODEL_DISPLAY_NAME \
  --container-image-uri="$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY_NAME/$IMAGE_NAME" \
  --container-ports=8080 \
  --container-predict-route="/predict" \
  --container-health-route="/health"
    
  
# create endpoint
gcloud ai endpoints create \
  --project=$PROJECT_ID \
  --region=$LOCATION \
  --display-name=$ENDPOINT_DISPLAY_NAME

# check url in vertex ai in model registry/online prediction
MODEL_ID="000000000000000000"
ENDPOINT_ID="000000000000000000"

# deploy model to endpoint
gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --project=$PROJECT_ID \
  --region=$LOCATION \
  --model=$MODEL_ID \
  --traffic-split=0=100 \
  --machine-type="n1-standard-2" \
  --display-name=$DEPLOYMENT_NAME \
  --enable-access-logging