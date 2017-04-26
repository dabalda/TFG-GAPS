function [ v, Q ] = bellmanOperatorV( problem, v_old, PI )
%BELLMANOPERATORV Summary of this function goes here
%   [ v, Q ] = bellmanOperatorV( problem, v_old, PI )

Q = getQfromV(problem, v_old);
v = getVfromQ(PI, Q);

end

