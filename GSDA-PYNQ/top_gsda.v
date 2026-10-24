module top_gsda (
    input wire clk_in,
    input wire reset_in
);
    wire clk = clk_in;
    wire reset;
    wire signed [7:0] dividend;
    wire signed [3:0] divisor;
    wire signed [7:0] quotient;

    wire [0:0] vio_reset_out;
    wire [7:0] vio_dividend_out;
    wire [3:0] vio_divisor_out;
    wire [7:0] vio_quotient_in;

    // Assignments
    assign reset           = vio_reset_out[0] | reset_in;
    assign dividend        = vio_dividend_out;
    assign divisor         = vio_divisor_out;
    assign vio_quotient_in = quotient;

    // Instantiations
    gsda u_gsda (
        .clk      (clk),
        .reset    (reset),
        .dividend (dividend),
        .divisor  (divisor),
        .quotient (quotient)
    );

    vio_0 u_vio (
        .clk        (clk),
        .probe_in0  (vio_quotient_in),   // 8-bit quotient monitor
        .probe_out0 (vio_reset_out),     // 1-bit reset control
        .probe_out1 (vio_dividend_out),  // 8-bit dividend control
        .probe_out2 (vio_divisor_out)    // 4-bit divisor control
    );

endmodule