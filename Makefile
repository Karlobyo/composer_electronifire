#################### PACKAGE ACTIONS ###################

reinstall_package:
	@pip uninstall -y composer_electronifire || :
	@pip install -e .

upload_data:
	gcloud config set project ${PROJECT}
	bq mk ${DATASET_NAME}
	bq load --autodetect --source_format CSV ${DATASET_NAME}.${FEATURES_NAME} ${LOCAL_DATA_PATH}/${FEATURES_NAME}.csv
	bq load --autodetect --source_format CSV ${DATASET_NAME}.${TARGET_NAME} ${LOCAL_DATA_PATH}/${TARGET_NAME}.csv

run_com_el_train:
	python -c 'from composer_electronifire.interface.main import run_model_training; run_model_training()'

start_instance:
	gcloud compute instances start ${INSTANCE}

connect_instance:
	gcloud compute ssh ${INSTANCE}

upload_package:
	gcloud compute scp --recurse ${LOCAL_PROJECT_PATH} ${INSTANCE}:~

download_results:
	gcloud compute scp --recurse ${INSTANCE}:${INSTANCE_REGISTRY_PATH} ${LOCAL_PROJECT_PATH}/${LOCAL_PROJECT_FOLDER}
