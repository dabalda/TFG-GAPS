function [ v ] = getVfromQ( PI, Q )
%GETVFROMQ Get v(s) from Q(s,a).
%   [ v ] = getVfromQ( PI, Q )

v = sum(PI.*Q,2);
end

