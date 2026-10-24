# Goldschmidt Division Algorithm (GSDA) on PYNQ FPGA

## 1. Project Overview

This project implements the **Goldschmidt Division Algorithm (GSDA)** using Verilog HDL and targets a Xilinx PYNQ FPGA board. Vivado's Virtual Input/Output (VIO) IP is used to provide input values and monitor the computed quotient through the JTAG interface.

The objective is to implement hardware-based division using fixed-point arithmetic and iterative multiplication instead of a conventional combinational divider.

## 2. Objectives

- Implement the Goldschmidt division algorithm using Verilog HDL.
- Use fixed-point arithmetic to represent intermediate values.
- Normalize the divisor before iterative computation.
- Implement the controller using a finite-state machine (FSM).
- Use Vivado VIO to control the inputs and observe the quotient.
- Verify the design through simulation and FPGA hardware testing.

## 3. Tools and Technologies

- **Hardware:** Xilinx PYNQ FPGA board
- **HDL:** Verilog
- **Development environment:** Xilinx Vivado 2025.1
- **Hardware debugging:** Vivado VIO IP
- **Programming and debugging interface:** JTAG
- **Simulation:** Vivado Simulator, if used for verification

## 4. Architecture

The design consists of two principal modules.

### `gsda.v`

Implements the division algorithm and contains:

- Sign detection and absolute-value calculation
- Fixed-point register initialization
- Divisor normalization
- Goldschmidt factor calculation
- Iterative multiplication and register updates
- Quotient extraction

### `top_gsda.v`

Acts as the top-level module and connects:

- The clock and reset inputs
- The GSDA computation module
- VIO output probes for dividend, divisor, and reset
- The VIO input probe for quotient monitoring

## 5. Algorithm

The Goldschmidt method approximates division by repeatedly updating the numerator and denominator using a common factor.

For numerator \(N\) and denominator \(D\), the iteration is:

\[
F_i=2-D_i
\]

\[
N_{i+1}=N_iF_i
\]

\[
D_{i+1}=D_iF_i
\]

As the denominator converges towards 1, the numerator converges towards the quotient.

The implementation uses a 16.16 fixed-point representation for intermediate registers. The upper 16 bits represent the integer portion and the lower 16 bits represent the fractional portion.

## 6. Hardware Interaction

VIO allows the user to enter the dividend and divisor and monitor the quotient directly through Vivado Hardware Manager.

This approach avoids the need for additional physical switches and LEDs for the input and output interface. The FPGA design is accessed through JTAG, and the VIO probes provide the interactive debugging interface.

## 7. Repository Structure

The constraint file, VIO IP file, testbench, and screenshot should be included only when available and applicable to the project.

## 8. Setup and Execution

1. Open Xilinx Vivado 2025.1.
2. Create a project for the exact target PYNQ FPGA board.
3. Add `gsda.v` and `top_gsda.v` as design sources.
4. Add `tb_gsda.v` as a simulation source, if available.
5. Import or recreate the VIO IP with the appropriate probe widths.
6. Add the correct board-specific XDC constraints.
7. Set `top_gsda` as the top module.
8. Run synthesis and implementation.
9. Generate the bitstream.
10. Open Hardware Manager and connect to the FPGA through JTAG.
11. Program the FPGA and open the VIO dashboard.
12. Apply input values and observe the quotient.

## 9. Limitations and Future Improvements

- Improve fixed-point precision and rounding behavior.
- Add explicit completion and divide-by-zero status signals.
- Verify signed division, overflow, and boundary cases.
- Measure FPGA resource utilization and timing.
- Compare the implementation against a reference software calculation.
- Optimize the number of iterations and hardware resource usage.

## 10. Conclusion

This project demonstrates an FPGA implementation of iterative division using the Goldschmidt algorithm. Verilog describes the computational hardware, while Vivado provides synthesis, implementation, bitstream generation, and VIO-based hardware interaction.

**Note:** The implementation should be validated against a reference divider over a range of signed and unsigned test cases before claiming complete functional correctness.
