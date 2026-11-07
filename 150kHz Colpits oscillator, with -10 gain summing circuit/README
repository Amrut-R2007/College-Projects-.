# Colpitts Oscillator (150 kHz) and Summing Amplifier (Gain -10)

**RV College of Engineering, Bangalore**  
**Department of Electronics and Communication Engineering**  
**Course:** Analysis And Design of Digital Circuits Lab PBL (121EC81P) | **Year:** 2025-26

---

## 📑 Abstract
This project focuses on the design and hardware implementation of a **Colpitts oscillator operating at 150 kHz**, followed by an **inverting summing amplifier with a voltage gain of −10**. The oscillator generates a stable sinusoidal waveform using an LC tank circuit and an NPN transistor operating under the Barkhausen criterion. The generated signal is then fed to a summing amplifier to scale and condition the output. 

The circuit's performance was verified through software simulation and hardware implementation. Parameters such as oscillation frequency, output amplitude, and distortion were successfully measured and analyzed. This project demonstrates fundamental concepts of feedback, frequency stability, and signal conditioning in analog circuit design.

---

## 🎯 Problem Statement
Implement a Colpitts oscillator at 150 kHz, then feed its output to a summing amplifier with gain –10. Measure: Oscillation frequency, summed/amplified output, and distortion.

### Core Challenges:
1. **Oscillation Stability:** Designing the Colpitts circuit to achieve stable, sinusoidal oscillation exactly at 150 kHz, ensuring the BJT provides sufficient gain to sustain the oscillation.
2. **Signal Conditioning:** Accurately configuring a second-stage amplifier to apply a precise gain of $A_v = -10$ to the oscillator's output signal.

---

## 🛠️ Hardware & Software Requirements

### Software Tools
*   **LTSpice / Proteus:** For circuit simulation and testing prior to hardware implementation.

### Hardware Components
*   **Active Elements:** CL100S (or BC547/2N2222) NPN Medium Power Transistor, Op-Amp ICs (LM358 / LM741 / TL082).
*   **Passive Elements:** 
    *   Axial Inductor ($300 \mu\text{H}$)
    *   Capacitors (Ceramic & Electrolytic: $4.7\text{nF}, 3.3\text{nF}, 0.1\mu\text{F}, 1\mu\text{F}$)
    *   Assorted Resistors (for biasing and amplifier feedback)
*   **Lab Equipment:** DC Power Supply, Digital Multimeter (DMM), Breadboard, Oscilloscope (CRO/DSO), Function Generator, Jumper Wires.

---

## ⚙️ Methodology & Circuit Design

### 1. Colpitts Oscillator Design (150 kHz)
*   **Transistor Configuration:** NPN transistor with a Voltage Divider Biasing scheme for high stability against temperature and gain variations.
*   **Tank Circuit:** Consists of a $300 \mu\text{H}$ inductor and a capacitive divider ($C_1, C_2$). 
*   **Frequency Formula:** $f_o = \frac{1}{2\pi\sqrt{LC_{eq}}}$
*   **Feedback:** A feedback ratio of $C_2/C_1 \approx 4$ ensures robust feedback to the emitter, satisfying the Barkhausen Criterion for self-starting oscillations.

### 2. Inverting Summing Amplifier Design (Gain = -10)
*   **Op-Amp Configuration:** Inverting summing topology using LM358/LM741.
*   **Gain Formula:** $A_v = -\frac{R_f}{R_{in}} = -10$
*   **Function:** Increases the amplitude of the oscillator output while providing a $180^\circ$ phase shift.

### 3. Workflow
1. Requirement Analysis & Component Calculations
2. Circuit Simulation in LTSpice
3. Hardware Assembly on Breadboard
4. Data Analysis & Measurement (CRO/DSO Output Verification)
5. Documentation

---

## 📊 Results & Observations
*   **Simulation (LTSpice):** The simulated circuit successfully demonstrated self-starting sustained oscillations. The measured frequency in the simulation was closely aligned with the theoretical target (~$147.18 \text{ kHz}$).
*   **Hardware Implementation:** 
    *   The breadboard implementation yielded a stable sinusoidal waveform continuously.
    *   The summing amplifier accurately scaled the output with the target gain of $-10$.
    *   Signal purity was maintained with minimal distortion observed on the DSO.

---


**Project Guides:** 
*   Dr. Harsha (Assistant Professor)
*   Dr. Srividya P / Sreevidya P. (Associate/Assistant Professor)

---

## 📁 Repository Structure
*   `Simulations/`: Contains the LTSpice (`.asc`) simulation files.
*   `Documents/`: Contains the final project report (`AMC_pbl_report.docx`).
*   `Presentations/`: Contains Phase I and Phase III PPTs, as well as the digital poster (`AMC_Digital Poster.pptx`).

---

## 🙏 Acknowledgements
We thank the respected HOD and the supportive faculty of the Electronics and Communication Department at RV College of Engineering for their immense support and guidance in the completion of this project.

*“Go, change the world”* 🌍
