function [ PI, Q, v, n_it ] = PIq( problem, epsilon, PI_ini )
%PIQ Policy Iteration for state-action value functions.
%   [ PI, v, Q, n_it ] = PIv( problem, epsilon, PI_ini ) finds
%   optimal policy PI and state-action value vector for the problem. Policy
%   evaluation loop ends when delta < epsilon. The initial policy is PI_ini
%   or the random policy if PI_ini = []. Greedy policies select all actions
%   whose value is not worse than the best minus epsilon.

% Initialize PI to random policy if none is supplied
PI = PI_ini;
if isequal(PI, [])
    PI = problem.getRandomPolicy();
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