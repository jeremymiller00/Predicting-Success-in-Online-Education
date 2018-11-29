
-- copy csv files into table (first one done)
COPY studentregistrations
FROM '/Users/jeremymiller/GoogleDrive/Data_Science/Galvanize/dsi-capstone/data/raw/studentRegistrations.csv' WITH (FORMAT CSV, HEADER, FORCE_NULL(date_unregistration,date_registration));

COPY studentassessment
FROM '/Users/jeremymiller/GoogleDrive/Data_Science/Galvanize/dsi-capstone/data/raw/studentAssessment.csv' WITH (FORMAT CSV, HEADER, FORCE_NULL(score));

COPY studentvle
FROM '/Users/jeremymiller/GoogleDrive/Data_Science/Galvanize/dsi-capstone/data/raw/studentVle.csv' WITH (FORMAT CSV, HEADER);

COPY studentinfo
FROM '/Users/jeremymiller/GoogleDrive/Data_Science/Galvanize/dsi-capstone/data/raw/studentInfo.csv' WITH (FORMAT CSV, HEADER);

COPY assessments
FROM '/Users/jeremymiller/GoogleDrive/Data_Science/Galvanize/dsi-capstone/data/raw/assessments.csv' WITH (FORMAT CSV, HEADER, FORCE_NULL(date));

COPY courses
FROM '/Users/jeremymiller/GoogleDrive/Data_Science/Galvanize/dsi-capstone/data/raw/courses.csv' WITH (FORMAT CSV, HEADER);

COPY vle
FROM '/Users/jeremymiller/GoogleDrive/Data_Science/Galvanize/dsi-capstone/data/raw/vle.csv' WITH (FORMAT CSV, HEADER, FORCE_NULL(week_from,week_to));

