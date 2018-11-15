# Capstone Project Plan


## Basic Research Nov 7-8
Review literature:


### Note from reading dataset info:
Calculating final grades requires making some assumptions about assessment weights - weight of the assessment in %. Typically, Exams are treated separately and have the weight 100%; the sum of all other assessments is 100%. But we do have the weights!

B and J presentations should be analyzed separately as they may be different in stucture 

Nevertheless, for some presentations the corresponding previous B/J presentation do not exist and therefore the J presentation must be used to inform the B presentation or vice versa. In the dataset this is the case of CCC, EEE and GGG modules.

Some of the estimated final scores are above 100 with many nearing 200, indicating that some of the modules are 'double-modules'. I was able to determine which model/presentation combinations were double modules with the following method:

* Determine that all students with estimated final scores above 100 were in a presentation of module DDD. Among students with estimated final scores above 100, DDD was the only module present.

* Create a dataframe with this subset (student with a score over 100 in module DDD) with a columns for each presentation. Analyze the value counts of each presentation to see that the only presentations with values greater than zero are 2013J and 2014B, and 2013B.

* Conclude that module DDD presentation 2013J, module DDD presentation 2013B, and module DDD presentation 2014B were double modules. Go back and moltiply the estimated final scores for those modules by 0.5



### Questions about data set
Are the courses undergraduate or postgraduate?

Looks like studentVLE has row for every student for every date they logged in for each module? If I aggregate I lose the fidelity?
Same with student assessment? -- Feature engineering to the rescue.

How many assessment are there for each student in each module / presentation?
It varies, but each student has at least one assessment.

## Plot univariate and multivariate summaries of the data
to plot: 
student info: done

student registration: done

student assessment: done
Weak correlations between feaures overall

assessments:
TMA more left tailed: median around 80
Exams more right tailed: median around 70

vle:
more clicks for 'resource' by far
runner-up categories: oucontent, url, subpage

do more joint plots when big table assembled!
clicks vs ...
target vs ...
---

Observations:
Student info: disabability looks less likely to complete, presentation may have an effect, 

Outliers:


Missing data:
56 of the assessments have weight zero! 25% news flash-these are formative assessments
dates have some nan values, 5%


## Data merging and prep Nov7-10
Aggregate other features into student info table
Start simple and build up
Have big table by Nov 10

## Inputs and Outputs of model Defined
Features: (+ means 'need to add')
date registered

Clicks per day of module
Percent of module days VLE accessed
Number of days VLE asccessed
Max clicks in one day
+ days since last interaction?

Mean assessment score (thus far for new data points)
Days assessment submitted early
estimated final module score
score on first assessment
module
presentation
gender
region
highest_education
imd_band
age_band
disability

** Target(s) **
Module not completed
Final result
Estimated final grade


## Split data

## Develop pipeline



## Minumum Viable Product Due Nov 16


## Code Freeze Nov 26
