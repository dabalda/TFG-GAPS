function [ v, Q ] = bellmanOperatorV( problem, v_old, PI )
%BELLMANOPERATORV Bellman operator for v.
%   [ v, Q ] = bellmanOperatorV( problem, v_old, PI )
%   Bellman operator for policy PI.
%
%   [ v, Q ] = bellmanOperatorV( problem, v_old )
%   Optimal Bellman operator.

narginchk(2,3);

Q = getQfromV(problem, v_old);
if nargin < 3
    PI = problem.getGreedyPolicy(Q,0); % Tolerance = 0
end
v = getVfromQ(PI, Q);
end

