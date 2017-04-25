function [ PI, v, Q, n_it ] = PIv( problem, epsilon, PI_ini )
%PIV Policy Iteration for state value functions.
%   [ PI, v, Q, n_it ] = PIv( problem, epsilon, PI_ini ) finds
%   optimal policy PI and state value vector for the problem. Policy
%   evaluation loop ends when delta < epsilon. The initial policy is PI_ini
%   or the random policy if PI_ini = []. Greedy policies select all actions
%   whose value is not worse than the best minus epsilon.

% Get parameters
n_states =  problem.n_states;
n_actions = problem.n_actions;
gamma =     problem.gamma;
P =         problem.Pssa;
R =         problem.Rssa;

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
    v = PEv(problem, PI, epsilon);
    
    policy_stable = true;
    
    % Policy improvement
    PI_temp = PI;
    
    % Repeat v for each initial state and action
    v_ssa = repmat(v(:)',[n_states,1,n_actions]);
    % Partial result
    v_2 = R+gamma.*v_ssa;
    % Get Q
    Q = squeeze(sum(P.*v_2, 2));
    
    % Find best policy and update the existing one
    PI = problem.getGreedyPolicy(Q, epsilon); % Using epsilon as tolerance
    
    % End loop if policy no longer changes
    if ~isequal(PI_temp, PI)
        policy_stable = false;
    end
end