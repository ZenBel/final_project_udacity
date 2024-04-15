# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Z Belligoli created the model. It is logistic regression using the default hyperparameters in scikit-learn 1.3.2.

## Intended Use

This model should be used to predict whether the salary of a person exceeds 50k$ based off a handful of attributes. The users are government officials.

## Training Data

Information about the raw data can be found [here](https://archive.ics.uci.edu/dataset/20/census+income). The data was cleaned by removing samples with unkown or empty fields.

The original data set has 32562 rows. The cleaned dataset has 30162 rows. 80% of the cleaned dataset was used for training. No stratification was done. To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the labels.

## Evaluation Data

The remaining 20% of the cleaned dataset was used for testing.

## Metrics
Precision: 0.7081081081081081
Recall: 0.25804333552199604
F1: 0.37824831568816164

## Ethical Considerations

This model should not be used for any high-risk decision making. The model was trained on a relatively small amount of data and certain subgroups may be underrepresented.

## Caveats and Recommendations

The model was trained on census data from 1994 which may not generalize well to the current population. Regular retraining is recommended.
