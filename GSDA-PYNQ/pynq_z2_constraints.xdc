# Clock (adjust pin based on your setup; Y9 is an example for Pynq Z2 clock)
create_clock -period 8.000 -name clk_in -waveform {0.000 4.000} [get_ports clk_in]
set_property PACKAGE_PIN Y9 [get_ports clk_in]
set_property IOSTANDARD LVCMOS33 [get_ports clk_in]

# Optional external reset (e.g., on a button pin like BTN0 on R17)
set_property PACKAGE_PIN R17 [get_ports reset_in]
set_property IOSTANDARD LVCMOS33 [get_ports reset_in]