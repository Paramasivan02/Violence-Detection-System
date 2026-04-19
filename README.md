# Violence-Detection-System
An automated visual surveillance system designed to detect human anomalies (fights and falls) using pose estimation and machine learning. This system transforms passive CCTV monitoring into an active security solution by providing real-time email alerts upon detection.
# Overview
Traditional surveillance relies heavily on human attention, which degrades rapidly over time. This project implements a machine learning pipeline that:
1. **Extracts** human skeletal landmarks using Google's MediaPipe.
2. **Classifies** movement patterns using a Random Forest classifier.
3. **Notifies** security personnel immediately via automated email alerts with attached incident snapshots.
# Key Features
- **Human Pose Estimation:** Utilizes 33 3D skeletal landmarks for precise movement analysis.
- **Intelligent Classification:** Distinguishes between "Normal" activity and "Abnormal" incidents (falls/fights).
- **Automated Alerting:** Integrated SMTP protocol for instant email notifications.**
- **Interactive Dashboard:** A clean Streamlit web interface for video processing and alert management.
- **Data-Centric Design:** Modular scripts for custom data collection and model retraining.

# Methodology
The system follows a linear pipeline:
1. **Frame Capture:** Video frames are read via OpenCV.**
2. **Landmark Extraction:** MediaPipe identifies $(x, y)$ coordinates for 33 body joints.
3. **Feature Vectorization:** Landmarks are flattened into a 66-feature vector.
4. **Inference:** The vector is passed to the trained XGBoost model.
5. **Logic Trigger:** If prediction == 1, a cooldown timer is checked, an image is saved to /alerts, and an email is dispatched.

Author: Paramasivan A

Contact: [paramasivana02@gmail.com]
