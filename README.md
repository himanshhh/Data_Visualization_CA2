# 🧵 Productivity of Garment Employees – Shiny App Dashboard

## 📁 Project Overview

This project presents an **interactive multi-page Shiny dashboard** built using **VS Code and Shiny for Python**, which visualizes and explores the **productivity of garment workers** over time. It is based on the [Productivity Prediction of Garment Employees Dataset](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees) from the UCI Machine Learning Repository.

The app helps users understand productivity trends across various dimensions such as **departments, quarters, teams, incentives, idle time**, and more. It also supports **principal component analysis (PCA)** and **cluster analysis** to aid further modeling and segmentation efforts.

📍 **Live App:** [https://himansh.shinyapps.io/ca-himansh-shinyappfinal/](https://himansh.shinyapps.io/ca-himansh-shinyappfinal/)

---

## 🧮 Technologies Used

| Tool      | Purpose                                 |
|-----------|------------------------------------------|
| Python    | Data processing, dashboard development   |
| Shiny for Python | Multi-page web app/dashboard      |
| Plotly    | Interactive visualizations               |
| Pandas    | Data manipulation                        |
| Sklearn   | PCA and clustering                       |
| VS Code   | Development environment                  |

---

## 🧠 Problem Statement

The primary objective is to analyze **actual productivity vs. targeted productivity** of garment workers and to uncover:
- Patterns and outliers across teams and departments
- Influence of operational variables like **SMV**, **WIP**, **incentives**, and **overtime**
- Daily and weekly fluctuations in productivity
- Potential clusters of similar productivity behavior using PCA & k-means

---


## 📊 Dashboard Structure & Visuals

This interactive dashboard is organized into **multiple pages** for a structured, user-friendly experience.

### 🔹 Productivity Analysis Overview

Shows overall distribution of **actual vs. targeted productivity** using histograms and line charts.

![Overview Page](GWP1.png)

---

### 🔹 Department & Team Performance

Visualizes **department-wise and team-wise productivity trends**, including SMV, idle time, and incentive comparisons.

![Department Page](GWP2.png)

---

### 🔹 Time-based Insights

Allows filtering by **day of the week** and **quarter**, helping uncover operational bottlenecks or high-performance timeframes.

![Time Insights Page](GWP3.png)

---

### 🔹 PCA & Cluster Analysis

Applies **Principal Component Analysis (PCA)** and **k-means clustering** to find behavioral patterns across workers.

![PCA Clustering Page](GWP4.png)

---

## 📂 Dataset Summary

**Source:** [UCI ML Repository – Productivity of Garment Employees](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)

**Features (13 columns):**
- `date`, `day`, `quarter`, `department`, `team`
- `smv` (Standard Minutes Value), `wip` (Work In Progress), `over_time`, `incentive`
- `idle_time`, `idle_men`
- `actual_productivity`, `targeted_productivity` (target label)

**Cleaning & Preprocessing:**
- Fixed categorical typos (`sweing` → `sewing`)
- Removed missing values in `wip`
- Converted `date` to datetime and encoded categorical variables
- Formatted numerical variables for visual clarity

---

## 🔍 Key Insights

- **Department and day of the week** significantly impact actual productivity.
- **Incentives and overtime** do not linearly correlate with productivity improvements.
- **Idle time** and **team assignment** play a key role in variance of output.
- PCA + k-means segmentation reveals **distinct groups** of worker behavior.

---

## ✅ Conclusion

This dashboard enables users to:
- Interactively explore the productivity landscape
- Uncover key productivity drivers across operational dimensions
- Use PCA and clustering to identify meaningful patterns
- Support data-informed decisions in workforce and production management

---

## 📌 How to Use

1. Visit the hosted app here:  
   👉 [**LIVE LINK**](https://himansh.shinyapps.io/ca-himansh-shinyappfinal/)
2. Select desired page (tabs at the top) to explore:
   - Overview
   - Department & Team
   - Time Insights
   - PCA & Clustering
3. Use dropdown filters to explore different dimensions interactively.
4. Clone the repo to run the app locally with VS Code + Python Shiny.

---

## 🛠️ Future Improvements

- Add filters to limit time range more accurately within dataset boundaries
- Include model predictions (e.g., regression or classification of productivity)
- Improve visual theme consistency across all plots

---

