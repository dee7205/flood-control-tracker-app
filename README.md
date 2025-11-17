# 🚨 PH Flood Control Projects Tracker

**Philippines Flood Control Projects Tracker** is a Streamlit-based interactive dashboard that promotes **transparency and accountability** in public infrastructure spending. Users can explore flood control project data, visualize key metrics, detect anomalies, and gain insights into project performance.

**Disclaimer:** This project was created solely for **educational purposes** as part of a school assignment. It is **not an official government tool**.

---

## Features

- **Red Flag Metrics**  
  Highlights high-value projects, suspicious cost savings, delayed projects, and statistical anomalies.

- **Filters & Alerts**  
  Filter projects by **Region**, **Province**, and **Year**; customize alert thresholds for cost savings and project duration.

- **Interactive Visualizations**  
  - Cost analysis (histograms, scatter plots)  
  - Timeline analysis (project duration trends)  
  - Geographic mapping of projects  
  - Contractor performance dashboards  
  - Temporal trends (projects and costs over time)

- **Searchable Project Table**  
  Search by project description, contractor, region, or province.

- **Anomaly Detection**  
  Uses Isolation Forest to detect projects with unusual cost or duration patterns.

- **Disclaimer Handling**  
  Some provinces may appear under a wrong region due to inconsistencies in the source data.

---

## Installation

1. **Clone this repository**

```bash
git clone https://github.com/your-username/ph-flood-control-tracker.git
cd ph-flood-control-tracker
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

---
