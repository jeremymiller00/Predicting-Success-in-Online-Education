-- create student reg table
CREATE TABLE studentregistrations
(
    code_module VARCHAR(45),
    code_presentation VARCHAR(45),
    id_student INT,
    date_registration INT,
    date_unregistration INT,
    PRIMARY KEY(code_module, code_presentation, id_student)
);

-- create student assessment table
CREATE TABLE studentassessment
(
    id_assessment INT,
    id_student INT,
    date_submitted INT,
    is_banked INT,
    score FLOAT,
    PRIMARY KEY(id_student, id_assessment)
);

-- create student vle table
CREATE TABLE studentvle
(
    code_module VARCHAR(45),
    code_presentation VARCHAR(45),
    id_student INT,
    id_site INT,
    date INT,
    sum_click INT
);

CREATE TABLE studentinfo
(
    code_module VARCHAR(45),
    code_presentation VARCHAR(45),
    id_student INT,
    gender VARCHAR(3),
    region VARCHAR(45),
    highest_education VARCHAR(45),
    imd_band VARCHAR(16),
    age_band VARCHAR(16),
    num_of_prev_attempts INT,
    studied_credits INT,
    disability VARCHAR(3),
    final_result VARCHAR(45),
    PRIMARY KEY(code_module, code_presentation, id_student)
);

CREATE TABLE assessments
(
    code_module VARCHAR(45),
    code_presentation VARCHAR(45),
    id_assessment INT,
    assessment_type VARCHAR(45),
    date INT,
    weight FLOAT,
    PRIMARY KEY(code_module, code_presentation, id_assessment)
);

CREATE TABLE courses
(
    code_module VARCHAR(45),
    code_presentation VARCHAR(45),
    module_presentation_length INT
);

CREATE TABLE vle
(
    id_site INT,
    code_module VARCHAR(45),
    code_presentation VARCHAR(45),
    activity_type VARCHAR(45),
    week_from INT,
    week_to INT,
    PRIMARY KEY(code_module, code_presentation, id_site)
);
