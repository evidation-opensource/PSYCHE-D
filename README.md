# [PSYCHE-D: Predicting change in depression severity using person-generated health data](https://preprints.jmir.org/preprint/34148)

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)

PSYCHE-D (Predicting SeveritY CHangE in Depression) is a two-phase multi-class classification model that predicts longitudinal changes in depression severity using person-generated health data in the form of survey responses and consumer wearable sleep and step data. This repository contains code that can be used to replicate the results presented in ["PSYCHE-D: predicting change in depression severity using person-generated health data" DOI].

## Abstract

### Background:

In 2017, an estimated 17.3 million adults in the US experienced at least one major depressive episode, with 35% of them not receiving any treatment. Under-diagnosis of depression has been attributed to many reasons including stigma surrounding mental health, limited access to medical care or barriers due to cost.

### Objective:

To determine if low-burden personal health solutions, leveraging person-generated health data (PGHD), could represent a possible way to increase engagement and improve outcomes.

### Methods:

Here we present the development of PSYCHE-D (Prediction of SeveritY CHange - Depression), a predictive model developed using PGHD from more than 4000 individuals, that forecasts long-term increase in depression severity. PSYCHE-D uses a two-phase approach: the first phase supplements self-reports with intermediate generated labels; the second phase predicts changing status over a 3 month period, up to 2 months in advance. The two phases are implemented as a single pipeline in order to eliminate data leakage, and ensure results are generalizable.

### Results:

PSYCHE-D is composed of two Light Gradient Boosting Machine (LightGBM) algorithm-based classifiers that use a range of PGHD input features, including objective activity and sleep, self reported changes in lifestyle and medication, as well as generated intermediate observations of depression status. The approach generalizes to previously unseen participants to detect increase in depression severity over a 3-month interval with a sensitivity of 55.4% and a specificity of 65.3%, nearly tripling sensitivity, while maintaining specificity, versus a random model.

### Conclusions:

These results demonstrate that low-burden PGHD can be the basis of accurate and timely warnings that an individual's mental health may be deteriorating. We hope this work will serve as a basis for improved engagement and treatment of individuals suffering from depression. Clinical Trial: Data used to develop the model was derived from the Digital Signals in Chronic Pain (DiSCover) Project (Clintrials.gov identifier: NCT03421223)


### Dataset requirements

The input data required to generate 3-month training/testing samples for the model are:

- Screener and baseline survey responses: socio-demographic information, comorbidities and health status at enrollment
- Lifestyle and medical changes (LMC) survey responses: self-reported changes over the past month, including starting a new medication, increased physical activity or reduction in alcohol consumption
- PHQ-9 survey responses
- Wearable PGHD: sleep and step data from a consumer wearable, aggregated at day-level

The following figure summarizes the data required to generate a sample along with a timeline for data collection. For further details regarding the construction of a training/testing sample, please refer to our paper.

<p align="center">
  <img src="images/two_phase_model.png" width="700"/>
</p>

### Model construction

PSYCHE-D is a two phase approach to predicting changes in depression severity. The first phase consists of predicting PHQ-9 score category for a given month, and is [presented in the following work](https://dl.acm.org/doi/10.1145/3469266.3469878). The second phase of the model takes the generated intermediate PHQ-9 score categories as inputs, along with screener survey responses, LMC survey responses and wearable PGHD, as visualized in the diagram above.

## Getting started

### Prerequisites

To use this project, you must have Python 3.8 installed.

You must also have the following packages installed: NumPy, Pandas, scikit-learn, SciPy, statsmodels, XGBoost, LightGBM.

You can install the package prerequisites by running the following command: ```pip install -r requirements.txt```

To fit the model with your own data, you must prepare data that has been collected as described in the "Dataset description" section in a parquet file. If you would like to use the data described in the paper, then please visit [here](https://zenodo.org/record/5085146#.YPG_URMzYUE) to request access.

### Usage

To fit the PSYCHE-D model, run the following command:

```
python combined_pipeline.py --data_path DATA_PATH_HERE --output_path OUTPUT_PATH_HERE
```

## Paper

[PSYCHE-D: predicting change in depression severity using person-generated health data](https://preprints.jmir.org/preprint/34148)

## Citing

> Makhmutova M, Kainkaryam R, Ferreira M, Min J, Jaggi M, Clay I
> PSYCHE-D: predicting change in depression severity using person-generated health data
> JMIR Preprints. 08/10/2021:34148


