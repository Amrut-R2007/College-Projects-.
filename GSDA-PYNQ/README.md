# Goldschmidt Division Algorithm on PYNQ-Z2 FPGA

## 1. Project Overview

This project implements a **Goldschmidt Division Algorithm (GSDA)** using Verilog HDL and targets the PYNQ-Z2 FPGA development board.

The design performs division using fixed-point arithmetic and iterative multiplication. Vivado's **Virtual Input/Output (VIO)** IP core allows the user to provide the dividend and divisor and observe the calculated quotient without requiring additional physical switches or LEDs for all input and output bits.

The project is intended to demonstrate an iterative division algorithm, finite-state-machine control, fixed-point arithmetic, and FPGA-based hardware debugging.

## 2. Objectives

- Implement the Goldschmidt division algorithm using Verilog HDL.
- Perform division using fixed-point arithmetic.
- Handle positive and negative operands.
- Use iterative multiplication to estimate the quotient.
- Integrate the design with the PYNQ-Z2 FPGA board.
- Use Vivado VIO to control inputs and monitor the output.
- Verify the design through Verilog testbench simulation and FPGA hardware testing.

## 3. Hardware and Software Requirements

### Hardware
- PYNQ-Z2 FPGA development board
- USB programming connection
- Computer with a compatible operating system

### Software
- AMD Xilinx Vivado Design Suite 2025.1
- Verilog HDL
- Vivado Hardware Manager
- Vivado Virtual Input/Output (VIO) IP core

## 4. Project Files

The repository contains the following files:

| File | Description |
|---|---|
| `gsda.v` | Main Verilog module implementing the Goldschmidt division algorithm |
| `gsda_tb.v` | Testbench for verifying the division module through simulation |
| `top_gsda.v` | Top-level module connecting the GSDA module to the VIO IP |
| `gsda.xdc` | FPGA pin and timing constraints associated with the design |
| `pynq_z2_constraints.xdc` | PYNQ-Z2 board constraint definitions |
| `gsda 1.jpeg` | Project implementation or simulation screenshot |
| `gsda 2.jpeg` | Additional project implementation or simulation screenshot |
| `README.md` | Project documentation and usage instructions |

**Note:** The generated Vivado VIO IP configuration is part of the Vivado project. The repository should also include the VIO IP configuration or the instructions needed to regenerate it if the project is to be recreated on another computer.

## 5. Algorithm Description

Goldschmidt division converts division into a sequence of multiplication operations.

For a division operation,

\[
Q=\frac{N}{D}
\]

where \(N\) is the dividend and \(D\) is the divisor.

The algorithm iteratively updates the numerator and denominator using a common multiplier factor:

\[
F_i=2-D_i
\]

\[
N_{i+1}=N_iF_i
\]

\[
D_{i+1}=D_iF_i
\]

As the iterations progress, the normalized denominator approaches 1. Therefore, the updated numerator approaches the quotient.

The Verilog implementation uses fixed-point arithmetic to represent fractional values in hardware. The design includes operand sign handling, normalization, multiplier-factor calculation, iterative updates, and quotient extraction.

## 6. Design Architecture

The implementation consists of three main components.

1. **GSDA module (`gsda.v`):** Processes the operands, performs normalization and iterative multiplication, and calculates the quotient.
2. **Top-level module (`top_gsda.v`):** Connects the GSDA module to the clock, reset, and VIO interface.
3. **VIO IP (`vio_0`):** Provides virtual controls for the reset, dividend, and divisor, and monitors the quotient.

### VIO Interface

| Signal | Width | Function |
|---|---:|---|
| Reset | 1 bit | Resets the divider |
| Dividend | 8 bits | Selects the numerator |
| Divisor | 4 bits | Selects the denominator |
| Quotient | 8 bits | Displays the computed result |

The VIO interface reduces the need for physical input switches and output LEDs when testing multi-bit signals.

## 7. Implementation Procedure

1. Open AMD Xilinx Vivado Design Suite.
2. Create or open a project targeting the PYNQ-Z2 board.
3. Add `gsda.v` and `top_gsda.v` as design sources.
4. Add `gsda_tb.v` as a simulation source.
5. Add the appropriate XDC constraint files and verify that their pin assignments match the target board.
6. Add the VIO IP core and configure the probe widths:
   - `probe_out0`: 1 bit for reset
   - `probe_out1`: 8 bits for dividend
   - `probe_out2`: 4 bits for divisor
   - `probe_in0`: 8 bits for quotient
7. Connect the VIO and GSDA instances using `top_gsda.v`.
8. Set `top_gsda` as the synthesis top module.
9. Run behavioral simulation using `gsda_tb.v`.
10. Run synthesis and implementation.
11. Generate the bitstream.
12. Connect the PYNQ-Z2 board and open Vivado Hardware Manager.
13. Program the FPGA and open the VIO dashboard.
14. Set the dividend and divisor through VIO, apply reset when necessary, and observe the quotient.

## 8. Example Test Cases

The following are useful test cases for validating the design. The expected results are mathematical reference values; the actual hardware outputs must be verified.

| Dividend | Divisor | Expected quotient |
|---:|---:|---:|
| 25 | 5 | 5 |
| 64 | 4 | 16 |
| 15 | 3 | 5 |
| 26 | 5 | 5 |
| 20 | 6 | 3 |
| 10 | 0 | Division-by-zero handling |

For non-integer divisions, the expected output depends on the design's rounding or truncation method. The current top-level interface exposes only the quotient; it does not expose a separate completion or division-by-zero status signal.

## 9. Advantages of Using VIO

- Allows direct control of multi-bit input signals from Vivado Hardware Manager.
- Allows internal signals to be observed during FPGA operation.
- Reduces the need for external switches, displays, and additional interface circuitry.
- Simplifies debugging and testing during development.
- Enables convenient testing of multiple input combinations.

VIO is primarily a development and debugging interface. A standalone product would normally use an appropriate physical interface, such as switches, UART, or another communication interface.

## 10. Limitations and Future Improvements

- Expose a completion (`done`) signal and division-by-zero status.
- Improve fixed-point precision and rounding.
- Add automated test cases for boundary values and signed division.
- Verify behavior for the largest positive and negative operands.
- Measure resource utilization, maximum clock frequency, and latency.
- Compare the hardware implementation with a reference software model.
- Add a physical user interface or communication interface for standalone operation.

## 11. Repository Contents and Reproducibility

The Verilog source files and constraints provide the main implementation artifacts. The screenshots document the development process.

For complete reproducibility, consider adding the Vivado project configuration, a clear VIO configuration or regeneration procedure, simulation instructions, and verified hardware results. Do not commit generated build directories unnecessarily.

## 12. Conclusion

This project demonstrates an FPGA-based implementation of the Goldschmidt division algorithm using Verilog HDL, fixed-point arithmetic, and iterative multiplication. Integration with Vivado VIO enables flexible input control and quotient monitoring on the PYNQ-Z2 board, providing a practical platform for studying digital arithmetic hardware and FPGA debugging.

---

**Tools:** Verilog HDL, AMD Xilinx Vivado 2025.1, Vivado VIO IP, PYNQ-Z2 FPGA development board.

**Target platform:** PYNQ-Z2 FPGA.

**Project category:** Digital Design / FPGA-Based Arithmetic Unit.
