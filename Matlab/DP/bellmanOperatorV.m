function [ v, Q ] = bellmanOperatorV( problem, v_old, PI )
%BELLMANOPERATORV Bellman operator for v.
%   [ v, Q ] = bellmanOperatorV( problem, v_old, PI )
%   Optimal Bellman operator if PI = [].

Q = getQfromV(problem, v_old);
if isequal(PI, [])
    PI = problem.getGreedyPolicy(Q,0); % Tolerance = 0
end
v = getVfromQ(PI, Q);
end

