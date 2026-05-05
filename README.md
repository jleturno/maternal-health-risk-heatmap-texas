# Texas Maternal Health Risk Prioritization Dashboard

## Live App

[View the interactive dashboard](https://maternal-health-risk-heatmap-texas-sag9z8vmjjju5cktgl6eew.streamlit.app)

---

## Overview

A county-level maternal health analytics application designed to support evidence-based decision-making, risk prioritization, and equitable resource allocation across Texas.

Maternal health outcomes in the United States — and particularly in Texas — vary significantly across geography, access to care, and socioeconomic conditions. These disparities are often difficult to interpret quickly using raw data alone.

This project translates complex maternal health indicators into a structured, interpretable framework that highlights where risk is most concentrated and which factors may be contributing to that burden.

This tool is designed as a decision-support system for public health stakeholders, enabling identification of high-risk counties and key contributing factors to maternal health disparities.

---

## Why This Matters

Maternal health disparities are not randomly distributed. They reflect underlying differences in access to care, economic stability, chronic disease burden, and systemic inequities.

Without clear, accessible tools for interpreting these patterns:

- High-risk areas may go under-identified  
- Resources may not be allocated effectively  
- Critical intervention opportunities may be missed  

This dashboard is designed to make these patterns more visible, interpretable, and actionable.

---

## Key Features

- Interactive choropleth map of Texas counties  
- Composite maternal health risk score (percentile-based)  
- County-level risk profile with statewide ranking and context  
- Driver analysis comparing county values to Texas averages  
- Component comparison visualization across key metrics  
- Similarity analysis to identify comparable county profiles  
- Rankings for highest- and lowest-burden counties  
- Automated narrative summaries for communication and reporting  
- Methodology and interpretation guidance  

---

## Methodology

- All indicators are standardized into percentile-based values  
- Higher percentiles represent relatively higher burden or risk  
- A composite risk score aggregates multiple maternal health-related factors  
- Rankings are calculated across all Texas counties  

### Interpretation Notes

- This is a relative comparison framework, not an absolute risk model  
- Results should not be interpreted as causal  
- County-level data may mask within-county disparities  
- Findings should be used alongside domain expertise and local context  

---

## Use Cases

This tool is designed to support:

- Public health agencies identifying high-priority regions  
- Policy analysts evaluating geographic disparities  
- Health equity organizations targeting interventions  
- Hospital systems and planners assessing regional need  
- Nonprofits communicating risk patterns to stakeholders  

---

## Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Plotly  

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
