function [ Q, v ] = bellmanOperatorQ( problem, Q_old, PI )
%BELLMANOPERATORQ Summary of this function goes here
%   [ Q, v ] = bellmanOperatorQ( problem, Q_old, PI )

v = getVfromQ(PI, Q_old);
Q = getQfromV(problem, v);


end

