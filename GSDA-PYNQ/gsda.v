`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 20.01.2026 18:45:30
// Design Name: 
// Module Name: gsda
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module gsda(
    input  wire              clk,
    input  wire              reset,
    input  wire signed [7:0] dividend,
    input  wire signed [3:0] divisor,
    output reg  signed [7:0] quotient
);

    // 16.16 fixed-point registers
    reg signed [31:0] N_reg, D_reg, F_reg;
    reg signed [31:0] next_N, next_D;
    reg signed [63:0] mulN, mulD;

    reg [2:0] state;
    reg [2:0] iter;

    reg result_sign;
    reg [7:0] abs_dividend;
    reg [3:0] abs_divisor;

    // combinational multiply (Vivado safe)
    always @(*) begin
        mulN = N_reg * F_reg;
        mulD = D_reg * F_reg;
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state    <= 0;
            iter     <= 0;
            quotient <= 0;
            N_reg    <= 0;
            D_reg    <= 0;
            F_reg    <= 0;
        end else begin
            case (state)

                // --------------------
                // State 0: sign & abs
                // --------------------
                0: begin
                    iter <= 0;
                    if (divisor != 0) begin
                        result_sign <= dividend[7] ^ divisor[3];
                        abs_dividend <= dividend[7] ? -dividend : dividend;
                        abs_divisor  <= divisor[3]  ? -divisor  : divisor;
                        state <= 1;
                    end
                end

                // --------------------
                // State 1: load fixed
                // --------------------
                1: begin
                    N_reg <= {8'b0, abs_dividend, 16'b0};
                    D_reg <= {12'b0, abs_divisor, 16'b0};
                    state <= 2;
                end

                // --------------------
                // State 2: normalize
                // --------------------
                2: begin
                    if (D_reg > 32'h0001_8000) begin
                        D_reg <= D_reg >>> 1;
                        N_reg <= N_reg >>> 1;
                    end else if (D_reg < 32'h0000_8000) begin
                        D_reg <= D_reg <<< 1;
                        N_reg <= N_reg <<< 1;
                    end else begin
                        state <= 3;
                    end
                end

                // --------------------
                // State 3: F = 2 - D
                // --------------------
                3: begin
                    F_reg <= 32'h0002_0000 - D_reg;
                    state <= 4;
                end

                // --------------------
                // State 4: iterate
                // --------------------
                4: begin
                    next_N = mulN[47:16];
                    next_D = mulD[47:16];

                    N_reg <= next_N;
                    D_reg <= next_D;
                    iter  <= iter + 1;

                    if (iter == 3) begin
                        quotient <= result_sign ?
                                   -(next_N[23:16] + next_N[15]) :
                                    (next_N[23:16] + next_N[15]);
                        state <= 0;
                    end else begin
                        state <= 3;
                    end
                end
            endcase
        end
    end
endmodule
