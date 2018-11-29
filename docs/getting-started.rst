Getting started
===============

The data are available here: https://analyse.kmi.open.ac.uk/open_dataset

To get started, download the seven csv files and store them in a directorty titled data/raw. 

To build a postgresql database, execute the file data/create_database.sql to build the tables, then execute the file data/populate_database.sql to populate the tables from the csv files. 

To replicate and explore my results:

Run src/features/build_features.py

Run src/models/train_model_rf.py

Run src/models/predict_evaluate_model.py

Enjoy!
