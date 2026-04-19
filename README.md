# Violence-Detection-System
An automated visual surveillance system designed to detect human anomalies (fights and falls) using pose estimation and machine learning. This system transforms passive CCTV monitoring into an active security solution by providing real-time email alerts upon detection.
# Overview
Traditional surveillance relies heavily on human attention, which degrades rapidly over time. This project implements a machine learning pipeline that:
- **1.Extracts** human skeletal landmarks using Google's MediaPipe.
- **2.Classifies** movement patterns using a Random Forest classifier.
- **3.Notifies** security personnel immediately via automated email alerts with attached incident snapshots.
# Key Features
**Human Pose Estimation:** Utilizes 33 3D skeletal landmarks for precise movement analysis.
**Intelligent Classification:** Distinguishes between "Normal" activity and "Abnormal" incidents (falls/fights).
**Automated Alerting:** Integrated SMTP protocol for instant email notifications.**
**Interactive Dashboard:** A clean Streamlit web interface for video processing and alert management.
**Data-Centric Design:** Modular scripts for custom data collection and model retraining.

# Installation & Setup
# 1. Clone the Repository
git clone https://github.com/your-username/anomaly-detection-system.git
cd anomaly-detection-system
