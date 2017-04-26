function [ PI, Q, v, n_it ] = PIq( problem, epsilon, PI_ini )
%PIQ Policy Iteration for state-action value functions.
%   [ PI, Q, v, n_it ] = PIq( problem, epsilon, PI_ini ) finds
%   optimal policy PI and state-action value vector for the problem. Policy
%   evaluation loop ends when delta < epsilon. The initial policy is 
%   PI_ini. Greedy policies select all actions whose value is not worse 
%   than the best minus epsilon.
%
%   [ PI, Q, v, n_it ] = PIq( problem, epsilon ) same as above except than
%   the initial policy is the random policy.

narginchk(2,3);
% Initialize PI to random policy if none is supplied
if nargin < 3
    PI = problem.getRandomPolicy();
else
    PI = PI_ini;
end

% Initialize loop variables
policy_stable = false;
n_it = 0;
while ~policy_stable % Policy iteration main loop
    n_it = n_it+1;
    
    % Policy evaluation
    [Q, v] = PEq(problem, PI, epsilon);
    
    policy_stable = true;
    
    % Policy improvement
    PI_temp = PI;   
    
    % Find best policy and update the existing one
    PI = problem.getGreedyPolicy(Q, epsilon); % Using epsilon as tolerance
    
    % End loop if policy no longer changes
    if ~isequal(PI_temp, PI)
        policy_stable = false;
    end
end