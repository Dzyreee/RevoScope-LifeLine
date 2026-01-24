# RevoScope: Volunteer-Led AI Diagnostic Suite 🚑🩺

**LifeLines Hackathon 2026** | **Problem Statement:** HPS#3 - AI-Augmented Emergency Triage

## 🚩 The Problem: The "Expert Gap"
In high-casualty crisis events (earthquakes, floods, conflict), the ratio of patients to medical professionals is dangerously imbalanced. Doctors are often tied up with surgery or critical interventions, leaving a massive queue of unscreened patients. 

## 💡 The Solution: RevoScope
**RevoScope** is a force-multiplier for frontline clinics. It is a web-based (moving to mobile) diagnostic platform designed to be used by **volunteers or non-medical staff**. 

By using the RevoScope (hardware) and our AI analysis (software), a volunteer can perform a comprehensive 30-second heart-and-lung scan. This allows for rapid patient sorting (triage), ensuring that by the time a doctor is available, they already have a digital "heads-up" on the patient's condition.

---

## ✨ Key Capabilities

### 1. 🧠 AI Lung Pathology Detection
The software receives audio from the RevoScope and uses a Convolutional Neural Network (CNN) to detect patterns associated with:
- **Pneumonia & Infection:** (Fine/Coarse Crackles)
- **Asthma & Airway Distress:** (Wheezing)
- **Trauma-Induced Lung Collapse:** (Absence of breath sounds)
- **Bronchitis:** (Rhonchi)
- Etc.



### 2. 💓 Integrated Heart Rate (BPM)
Using digital signal processing, RevoScope extracts a patient's heart rate from the same recording:
- **Low-Pass Filtering:** Isolates the 20Hz–150Hz range to "hear" the heart through the lungs.
- **Peak Counting:** Calculates real-time BPM to detect shock or distress (Tachycardia).

### 3. 🚦 Automated Triage Ranking
The app provides an immediate, color-coded priority status:
- 🟢 **GREEN (Stable):** Normal vitals. Volunteer continues monitoring.
- 🟡 **YELLOW (Observation):** Minor issues detected (e.g., wheezing). Queue for doctor review.
- 🔴 **RED (Critical):** Life-threatening sounds or extreme heart rate. Immediate doctor intervention required.

---

## 🛠️ The Hardware: RevoScope
The RevoScope is an original, low-cost acoustic sensor designed for rugged use.
- **Components:** High-sensitivity capacitive microphone + 3D-printed/recycled rigid acoustic chamber.
- **Design Philosophy:** Optimized for digital sensors rather than human ears, allowing for better AI accuracy in noisy field hospitals.
- **Cost:** <$10 USD per unit, making it logistically feasible for mass distribution.

---

## 📁 Repository Structure
- `/app`: Logic flow and UI interface (Base44).
- `/hardware`: STL files and internal assembly diagrams for the RevoScope.
- `/model`: AI classification logic and frequency mapping documentation.
- `/test-samples`: Verified clinical audio files for Pneumonia, Wheezing, and Normal lungs.

---

## 🚀 Future Roadmap
- **Native Mobile App:** Direct smartphone sensor integration.
- **Cluster Mapping:** Real-time data visualization to help aid coordinators see "hot spots" of respiratory illness in camps.
- **Expert Remote Review:** Encrypted audio uploads for remote specialists.

