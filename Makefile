#################### PACKAGE ACTIONS ###################

reinstall_package:
	@pip uninstall -y composer_electronifire || :
	@pip install -e .

upload_data:
	gcloud config set project ${PROJECT}
	bq mk train_data
	bq load --autodetect --source_format CSV train_data.X ./preprocessed_data/X.csv
	bq load --autodetect --source_format CSV train_data.y ./preprocessed_data/y.csv

run_com_el_train:
	python -c 'from taxifare.interface.main import composer_electronifire.interface.main import run_com_el_train; run_com_el_train()

start_instance:
	gcloud compute instances start ${INSTANCE}

connect_instance:
	gcloud compute ssh ${INSTANCE}

upload_package:
	gcloud compute scp --recurse ${LOCAL_PROJECT_PATH} ${INSTANCE}:~

download_results:
	gcloud compute scp --recurse ${INSTANCE}:${INSTANCE_REGISTRY_PATH} ${LOCAL_PROJECT_PATH}/${LOCAL_PROJECT_FOLDER}
