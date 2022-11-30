#################### PACKAGE ACTIONS ###################

reinstall_package:
	@pip uninstall -y composer_electronifire || :
	@pip install -e .

run_preprocess:
	python -c 'from composer_electronifire.preprocessor.main import preprocess; preprocess(); preprocess(source_type="val")'

upload_data:
	gcloud config set project ${PROJECT}
	bq mk train_data
	bq load --autodetect --source_format CSV train_data.X ./preprocessed_data/X.csv
	bq load --autodetect --source_format CSV train_data.y ./preprocessed_data/y.csv

run_composer_electronifire:
	python -m composer_electronifire.interface.main
