function [ Q, v ] = bellmanOperatorQ( problem, Q_old, PI )
%BELLMANOPERATORQ Bellman operator for Q.
%   [ Q, v ] = bellmanOperatorQ( problem, Q_old, PI )
%   Bellman operator for policy PI.
%
%   [ Q, v ] = bellmanOperatorQ( problem, Q_old )
%   Optimal Bellman operator.

narginchk(2,3);

if nargin < 3
    PI = problem.getGreedyPolicy(Q_old,0); % Tolerance = 0
end
v = getVfromQ(PI, Q_old);
Q = getQfromV(problem, v);
end

