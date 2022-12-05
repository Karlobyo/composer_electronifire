# composer_electronifire
Electronify the big maestros from the past...

make commands of package:

reinstall_package - Uninstalls isnstalled version of composer_electronifire and
                    installs with -e parameter. exits with error if no version
                    is installed.

upload_data - Sets gcloud project to .env PROJECT. Creates dataset on google bigquery.
              Uploadst X and y to goolge storrage

run_com_el_train - Run training of the baseline model

run_mv_preprocess - Run preprocessing of raw data midi files

run_mv_train - Run training of multivariate model on preprocessed data

run_mv_predict - Run prediction for seed

start_instance - Starts instance. Only make from local!

connect_instance - Connects to instance. Only run from local!

upload_package - Uploads package. Only run from local!

download_results - Downloads results from virtual machine. Only run from local!
