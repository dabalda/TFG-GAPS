function [ v ] = getVfromQ( PI, Q )
%GETVFROMQ Summary of this function goes here
%   [ v ] = getVfromQ( PI, Q )

v = sum(PI.*Q,2);
end

