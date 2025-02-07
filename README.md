# BTTAI_AJL_Team_2
# Equitable AI for Dermatology
## Repo for Equitable AI for Dermatology Kraggle 2025 Competition

# Project Overview 
Skin condition classification models often exhibit bias due to the lack of diverse training data, leading to disparities in healthcare, particularly for individuals with darker skin tones. This project, part of the Spring 2025 AI Studio, aims to address this challenge by developing a machine learning model that classifies 16 different skin conditions across diverse skin tones while ensuring fairness and explainability.

The competition is hosted by Break Through Tech AI and the Algorithmic Justice League (AJL), with an emphasis on equity and transparency in AI-driven healthcare solutions.

# Dataset 
This project utilizes a subset of the Fitzpatrick17k dataset, a collection of 17,000 labeled dermatological images. The dataset includes:

* 4,500 images across 16 skin conditions

* Images labeled with Fitzpatrick skin tone scale (1-6), indicating skin type diversity

* Annotations for medical conditions sourced from DermaAmin and Atlas Dermatologico

| Column                | Data Type | Description                                              |
|-----------------------|----------|----------------------------------------------------------|
| `md5hash`            | Object   | Unique image identifier                                  |
| `fitzpatrick_scale`  | Int      | Self-reported Fitzpatrick skin tone (1-6)               |
| `fitzpatrick_centaur`| Int      | Expert-assigned Fitzpatrick skin tone                   |
| `label`              | Object   | Target skin condition label                             |
| `nine_partition_label`  | Object   | Grouped diagnostic category                            |
| `three_partition_label` | Object   | Broad diagnostic classification                        |
| `qc`                 | Object   | Quality control flag by board-certified dermatologists  |
