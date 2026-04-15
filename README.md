# Diabetes Risk Decision Support System

# Overview

This project was developed for BC Analytics, a health-tech startup aiming to improve patient outcomes through data-driven healthcare solutions. The goal is to identify patients at risk of diabetes early and support healthcare providers with actionable insights.

# Objectives

The project aims to:

Predict diabetes risk stage (diabetes_stage)
Identify lifestyle factors influencing diabetes risk
Segment patients based on health and lifestyle characteristics
Provide insights through an interactive dashboard
Methodology (CRISP-DM)

1. Business Understanding
Healthcare providers need tools to detect diabetes risk early and support decision-making.

2. Data Understanding
The Diabetes_and_Lifestyle_Dataset contains health and lifestyle variables used to analyse diabetes risk.

3. Data Preparation
Data cleaning, feature selection, encoding, and scaling were performed before modelling.

4. Modeling
Two models were implemented, namely:
    Random Forest (primary model)
    Decision Tree (comparison model)

# Risk Classification:

Decision Tree
Random Forest
XGBoost

Patient Segmentation:

K-Means Clustering (k = 3)

# Key Driver Analysis:

SHAP (Shapley Additive Explanations)
Web Application

An interactive DASH dashboard was developed to display predictions, lifestyle insights, and recommendations for healthcare providers.

# Deployment: Render
