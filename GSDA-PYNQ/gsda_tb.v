`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 20.01.2026 18:46:59
// Design Name: 
// Module Name: gsda_tb
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



module gsda_tb;

    reg clk;
    reg reset;
    reg signed [7:0] dividend;
    reg signed [3:0] divisor;
    wire signed [7:0] quotient;

    gsda uut (
        .clk(clk),
        .reset(reset),
        .dividend(dividend),
        .divisor(divisor),
        .quotient(quotient)
    );

    // 10 ns clock
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        reset = 1;
        dividend = 0;
        divisor  = 0;

        #20 reset = 0;

        // 64 / 4 = 16
        dividend =  8'sd64;
        divisor  =  4'sd4;
        #250;

        // 100 / 5 = 20
        dividend =  8'sd100;
        divisor  =  4'sd5;
        #250;

        // -100 / 5 = -20
        dividend = -8'sd100;
        divisor  =  4'sd5;
        #250;

        // 64 / -4 = -16
        dividend =  8'sd64;
        divisor  = -4'sd4;
        #250;

        $finish;
    end
endmodule