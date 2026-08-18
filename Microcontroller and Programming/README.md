# ♻️ Automated Smart Waste Management System

## 📌 Overview
This repository contains the firmware, design reports, and presentation materials for an Advanced Automated Waste Management System. Developed as part of the Electronics and Communication Engineering (ECE) Project-Based Learning (PBL) curriculum, this prototype moves beyond standard hobbyist designs to deliver a robust, power-efficient, and scalable sanitation solution. 

The system automates user detection, bin actuation, and internal fill-level monitoring to optimize waste collection cycles and minimize physical contact.

## 🧠 The Dual-MCU Architecture
A defining feature of this project is its **Dual Microcontroller Architecture**, which is why this repository contains both Keil (`.uvprojx`) and Arduino (`.ino`) codebases. 

Instead of forcing a single chip to handle every task sequentially, the workload is distributed:
* **Primary Core (STM32 Black Pill - Keil):** Acts as the "main brain." Utilizing the ARM Cortex-M4 processor, it handles heavy mathematical computations, precise hardware timers for non-blocking motor actuation, and complex real-time decision-making.
* **Secondary Core (Arduino-compatible MCU - .ino):** Acts as the auxiliary controller. It offloads secondary tasks (such as basic sensor polling, status indicators, or communication interfaces) to ensure the Black Pill’s main control loop is never delayed or interrupted. 

## 🚀 What Makes This Unique?
Most existing "smart dustbins" follow a very predictable formula: an 8-bit Arduino Uno running blocking `delay()` functions and a fragile hobby servo motor for the lid. Our approach redesigns this from the ground up:
1. **No Fragile Servo Motors:** Instead of relying on standard servo motors which easily strip their gears under heavy loads, this system utilizes a more robust motor-driver-based actuation mechanism, ensuring long-term mechanical reliability.
2. **Interrupt-Driven Logic:** The system operates purely on hardware interrupts and timers rather than blocking code, ensuring deterministic and instant reaction times when a user approaches.
3. **Distributed Processing:** The dual-MCU setup prevents bottlenecks. One MCU handles the physical mechanics while the other handles environmental monitoring, creating a highly stable system.

## 📐 Methodology
Our development followed a strict engineering design process:
1. **Data Acquisition:** Proximity and depth sensors continuously scan the environment. 
2. **Signal Filtering:** Raw sensor data is processed to remove noise (preventing the bin from opening randomly due to passing shadows or insects).
3. **State Machine Execution:** The Black Pill evaluates the filtered data using a Finite State Machine (FSM). It determines if the bin is full (locking the system and alerting staff) or empty enough to accept waste.
4. **Actuation:** The motor driver smoothly engages the actuation mechanism to open the lid, holding it open only while the user is present, before safely closing it.

## 🔋 Power Efficiency
In municipal or large-scale deployments, power consumption is a critical factor. This project was designed with efficiency in mind:
* **Low-Power Idle States:** The Cortex-M4 architecture allows the Black Pill to process sensor data incredibly fast and return to an idle state, saving energy compared to slower chips that must stay awake longer to do the same math.
* **Smart Polling:** Instead of continuously powering the sensors at maximum frequency, the system optimizes polling rates based on the current state (e.g., polling less frequently when the bin is known to be completely empty).
* **Zero Idle Motor Current:** The motor driver logic ensures that power to the actuation motors is completely cut when the lid is resting, preventing unnecessary battery drain and motor heating.

## 📁 Repository Contents
* **`MCP_PBL.uvprojx`**: The primary Keil uVision project containing the bare-metal C/C++ firmware for the STM32 Black Pill.
* **`MCP_PBL_UPDATED.ino`**: The Arduino IDE code for the secondary auxiliary microcontroller.
* **`MCP_PBL_Updated_Report.docx`**: The comprehensive technical report including block diagrams, schematics, and theoretical analysis.
* **`PBL ppt.pptx` & `Poster Template.pptx`**: Presentation materials and visual aids for the final project defense.

## 👨‍💻 Credits
* **Author:** Amrut R
* Developed for the Microcontrollers and Programming (MCP) PBL curriculum.
