# Composer Electronifire
Electronifire the big maestros from the past...

***

Descripition and purpose:

This is an app to generate new piano music based on midi datasets from specific classical composers. In a second step the app would automatically generate and apply an electronic beat on the newly generated piano music (this second step is still a work in progress though, manual at the moment...). 

It is thought as a source of creative enrichment and inspiration for musicians and music producers and an attempt to answer to all the people who always wondered: 'How would that great composer from the past sound like, if he would be alive nowadays?'

***

Installation:

Clone the repository and install the requirements listed in the requirements.txt file.
From your terminal go to the 'webinterace' folder containing the 'app_local.py' file and then run 'streamlit run app_local.py'

Tadaaa, that's it! You can click on a composer (only Chopin available at the moment), generate new piano music and even listen to an electronifired demo sample

***

Make commands of package:

reinstall_package - Uninstalls installed version of composer_electronifire and
                    installs with -e parameter. exits with error if no version
                    is installed.

upload_data - Sets gcloud project to .env PROJECT. Creates dataset on google bigquery.
              Uploads X and y to goolge storage

run_com_el_train - Run training of the baseline model

run_mv_preprocess - Run preprocessing of raw data midi files

run_mv_train - Run training of multivariate model on preprocessed data

run_mv_predict - Run prediction for seed

start_instance - Starts instance. Only make from local!

connect_instance - Connects to instance. Only run from local!

upload_package - Uploads package. Only run from local!

download_results - Downloads results from virtual machine. Only run from local!
