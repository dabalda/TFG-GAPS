function [ Q, v ] = bellmanOperatorQ( problem, Q_old, PI )
%BELLMANOPERATORQ Bellman operator for Q.
%   [ Q, v ] = bellmanOperatorQ( problem, Q_old, PI )
%   Optimal Bellman operator if PI = [].

if isequal(PI, [])
    PI = problem.getGreedyPolicy(Q_old,0); % Tolerance = 0
end
v = getVfromQ(PI, Q_old);
Q = getQfromV(problem, v);


end

