# Handling-Estimation-Analysis

A Generic approach to built a Classifier that will predict and estimate large amount of natural language data using "Natural Language Processing".

__Introduction__:

To process with any text of natural language either an article or a book, just add the data source name into

> __init__.py script " main_processing(file_name'required_Data_Set.type') "


After that, analyze the resultant ( 2 x 2 ) confusion matrix to give predictions. Here, confusion matrix gives for types of results
such as;


__General Stats__:


> 1- TRUE POSITIVE : measures the proportion of actual positives that are correctly identified.

> 2- TRUE NEGATIVE : measures the proportion of actual positives that are not correctly identified.

> 3- FALSE POSITIVE : measures the proportion of actual negatives that are correctly identified.

> 4- FALSE NEGATIVE : measures the proportion of actual negatives that are not correctly identified.


True or False refers to the assigned classification being Correct or Incorrect, while Positive or Negative refers to assignment to the Positive or the Negative Category;


__Proformance Vector__:


__texted_x=200__  |  __Pridicted NO__ |   __Predicted Yes__
--------------|---------------|----------------
Actual NO     |  True +ve     |   False -ve
Actual Yes    |  False +ve    |   True -ve
