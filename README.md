# composer_electronifire
Electronify the big maestros from the past...

make commands of package:

reinstall_package - Uninstalls isnstalled version of composer_electronifire and
                    installs with -e parameter. exits with error if no version
                    is installed.

upload_data - Sets gcloud project to .env PROJECT. Creates dataset on google bigquery.
              Uploadst X and y to goolge storrage

run_com_el_train - Run training of the model

start_instance - Starts instance. Only make from local!

connect_instance - Connects to instance. Only run from local!

upload_package - Uploads package. Only run from local!

download_results - Downloads results from virtual machine. Only run from local!
