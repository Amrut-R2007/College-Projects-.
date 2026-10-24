# Smart Sort: AI-Powered Rapid Grading System for Second-Life Li-ion Cells

## 📌 Project Overview
**Smart Sort** is a low-cost, AI-driven, rapid battery grading system designed to estimate the true **State of Health (SoH)** of discarded lithium-ion cells in a few seconds. Developed as part of the Design Thinking Lab (DTL) at RV College of Engineering, this project addresses the global electronic waste problem by identifying reusable batteries suitable for secondary (second-life) applications, such as solar energy storage, backup power systems, and DIY electronics.

Traditional capacity testing takes several hours per cell, and Electrochemical Impedance Spectroscopy (EIS) requires expensive lab equipment. **Smart Sort** solves this by analyzing the **dynamic voltage load-and-recovery response** of a battery when subjected to a short current pulse, extracting physics-based features, and classifying the cell using machine learning.

---

## 🛠️ System Architecture & Hardware Implementation
The prototype consists of two stages: a hardware measurement setup and a software-based AI classification pipeline.

### Hardware Components:
*   **Microcontroller:** Arduino Uno (controls the pulse timing sequence and transmits serial data).
*   **ADC:** ADS1115 16-bit Analog-to-Digital Converter (provides high-precision voltage monitoring over $I^2C$).
*   **Constant Current Load Regulator:** LM358 Operational Amplifier regulating an **IRLZ44N MOSFET** to draw a steady ~1 A load current.
*   **Sensing Resistor:** 1 $\Omega$ power shunt resistor for current validation.
*   **Sorting Mechanism:** Prototype sorting chute powered by servo motors to physically separate cells.

### Operation Sequence:
1.  **Open-Circuit State:** The system measures the battery's initial resting voltage.
2.  **Constant-Current Load Pulse:** A steady 1 A load is applied for a short duration, causing an immediate voltage sag governed by internal resistance.
3.  **Recovery State:** The load is cut off, and the battery voltage gradually recovers back to its equilibrium state.
4.  **Data Transmission:** Arduino streams the high-precision voltage readings to the host PC via serial connection.

---

## 🧠 Feature Extraction & Machine Learning Pipeline
Instead of feeding raw data directly, the system extracts four physics-informed diagnostic features from the dynamic load-and-recovery voltage curve:

1.  **Internal Resistance ($R$):** Calculated using Ohm's Law ($R = \Delta V / I$) from the initial instantaneous voltage drop.
2.  **Maximum Voltage Drop:** The difference between the initial open-circuit voltage and the minimum voltage observed during the load pulse.
3.  **Recovery Amount:** The total voltage recovered after the load is removed.
4.  **Initial Recovery Slope:** The rate of voltage recovery immediately after load cutoff. This feature reflects battery polarization and ion diffusion dynamics, serving as the most discriminative feature for classification.

### Machine Learning Classifier:
*   **Preprocessing:** Features are standardized using `StandardScaler`.
*   **Model:** A **Random Forest Classifier** implemented in Python using the `scikit-learn` library. Random Forest was selected due to the highly non-linear nature of battery degradation and the structured, low-dimensional feature set.
*   **Training Data:** The model was trained and validated on a dataset of **300 Monte Carlo LTspice simulations** representing Grade A, Grade B, and Grade C battery conditions.
*   **Accuracy:** Achieved **100% classification accuracy** on the simulated test dataset.

---

## 📊 Battery Grading Criteria
*   **Grade A (Healthy):** Shallow voltage sag and fast, near-complete recovery. Indicates low internal resistance and healthy diffusion dynamics. Suitable for high-demand second-life applications.
*   **Grade B (Aged but Usable):** Moderate voltage sag and slower recovery. Suitable for low-to-moderate power storage applications.
*   **Grade C (End-of-Life):** Deep voltage sag and sluggish, incomplete recovery. Points to high internal resistance and degraded diffusion pathways. Unsuitable for reuse and marked for recycling.

---

## 🚀 Future Scope
*   **Real-World Validation:** Test and calibrate the classifier using physically aged lithium-ion battery datasets.
*   **Chemistry Generalization:** Expand training to support multiple chemistries (e.g., LFP, NMC), capacities, and manufacturers.
*   **Full Automation:** Integrate the automated physical sorting chute with the classification software to run high-throughput sorting without manual intervention.
*   **Cloud Integration:** Establish cloud storage for test logs to monitor battery recycling centers.

---

## 👥 Contributors
*   **Amrut R** (1RV24EC025)
*   **Anurag** (1RV24EC031)
*   **Bhavesh S** (1RV24EC044)
*   **Chandana B C** (1RV24EC053)

*Under the guidance of:*  
**Ms. R Sindhu Rajendran**, Assistant Professor, Dept. of ECE, RV College of Engineering.
