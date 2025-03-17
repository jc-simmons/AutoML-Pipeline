# AutoML Pipeline: Full Lifecycle Automation 

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies & Tools](#technologies--tools)
3. [Data Selection](#data-selection)
4. [Machine Learning](#machine-learning)
5. [Automation & Deployment](#automation--deployment)
6. [Disclaimer](#disclaimer)

---

## Project Overview

This repository contains files related to an ongoing project focused on the full cycle of data processing and machine learning. The goal is to create an environment to test and demonstrate various data science and ML tasks, specifically related to operations, mimicking real-world development workflows.

The current status of the project is as follows:

1. Source, clean, and model a regularly updated dataset.
2. Create a simple API to query predictions.
3. Containerize the application.
4. Deploy the container to a web host.
5. Automate the entire workflow, from data sourcing and processing to verifying the API.

---

## Technologies & Tools

- **Python** (for data processing, modeling, and Flask API)
- **Flask** (for building the web API)
- **Docker** (for containerization)
- **GitHub Actions** (for pipeline automation)
- **Render** (for hosting the API)
- **Docker/Docker Hub** (for containerizing and storing Docker images)
- **Pandas** (for data cleaning and manipulation)
- **Scikit-learn** (for machine learning)
- **Matplotlib/Seaborn** (for data visualization)

---

## Data Selection

The data used in this project are the monthly-updated StatCan Labour Force Survey Public Use Microdata Files. For practical reasons, only a single month of data is stored to keep the dataset manageable and avoid creating an overly large archive. This approach ensures compliance with the Statistics Canada Open License and helps maintain focus on relevant data without compromising data accessibility or storage requirements.

The data were selected primarily because:

1. They are regularly updated, requiring an adaptable and flexible approach.
2. The data is relatively messy (coded and survey-based), providing a challenging foundation for modeling.

---

## Machine Learning

The model is trained to predict hourly income based on demographic factors such as location, age, education, and more. A major focus of the machine learning aspect of the project is to construct a flexible and extensible Scikit-learn framework that can easily be applied to a variety of tasks.

Key components of the current implementation include:

- **Dynamic model imports**: All Scikit-learn models and testing/validation split generators are imported through a `config.yaml` file, allowing easy modification.
- **Data Loader class**: A class for adding different data splitting stages, making various validation and testing schemes easily accessible.
- **Model Runner class**: Extends Scikit-learn’s GridSearchCV, simplifying the process of training multiple models and hyperparameters.
- **Estimator class**: A wrapper for optional feature and target transformations with the model, similar to Scikit-learn's `Pipeline` estimator but more flexible, allowing for transformations like numerical encoding or squeeze methods.
- **Refitting**: Controlled via a flag in the config file, based on the best-performing model during validation.

### Model Results & Evaluation

Previous research on predicting alumni income through machine learning, as summarized in Table 1 of [Supervised Machine Learning Predictive Analytics for Alumni Income](https://doi.org/10.1186/s40537-022-00559-6) (2022), highlights the use of various performance metrics such as R², Pseudo-R², and multiclass classification accuracy. While this project isn't intended to be a rigorous academic study, it's helpful to compare its results with those in the literature. 

One key difference is that most studies in the review predict yearly income, whereas this project focuses on **hourly income**. Hourly income is more volatile, and this factor may influence model performance. Despite the increased variability of hourly income predictions, the best-performing model in this project achieved an **R² of 0.57**, slightly outperforming the highest result of **0.50** found in the linked review. This suggests that the model is capable of providing moderately strong predictions in this context.

For classification, the multiclass labels were determined by binning predicted income into quartiles (0-25%, 25-50%, 50-75%, and 75-100%), ensuring equal weighting. The model achieved a **multiclass classification accuracy of 0.63**, indicating a fairly balanced performance across the four income bins. This is comparable to the range of **0.38 to 0.84** observed in the studies reviewed, though it’s important to note that some of the studies had imbalanced classification tasks, making direct comparisons a bit challenging. As such, further investigation would be needed to fully assess multiclass classification performance in this case.

In conclusion, while the model's R² and multiclass accuracy scores are promising, they are not necessarily groundbreaking. More experimentation and tuning could improve the results, particularly as more data becomes available and as the model is refined further. Additionally, the choice of hourly income as the target variable may introduce challenges that are worth addressing in future work.

---

## Automation & Deployment

The automation pipeline is controlled using GitHub Actions workflows, consisting of two stages:

1. **data_pipeline**: This stage runs on a cron schedule or can be triggered manually. It checks the StatCan LFS archive for the most recent data. If the month and year match the current `data_version.json`, the workflow completes successfully. If there’s no match or the `data_version.json` file is missing, the full pipeline runs. This stage may be turned off at times to conserve resources.
   
2. **deploy_pipeline**: This stage triggers either manually or when a push to the `output/model.joblib` file (the final model) occurs. The model predictions are accessed via a simple Flask API, which is Dockerized, pushed to Docker Hub, deployed using a webhook trigger, and verified with a `test_sample.json` to ensure successful deployment with the correct HTTP status code.

Please note that access to the API is restricted via an authorization token, and the Render web service may require some time to spin up after a period of inactivity.

You can find more details on the automation in the `.github/workflows/main.yml` file.

---

## Disclaimer

This project uses data provided by Statistics Canada under the open license. The data is reproduced and distributed "as is" with the permission of Statistics Canada. This project does not imply endorsement by Statistics Canada, and the views expressed are those of the author(s), not Statistics Canada. All usage of the data complies with the terms outlined in the Statistics Canada Open License.




