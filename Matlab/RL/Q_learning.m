function [ PI, Q, episodes_count ] = Q_learning( problem, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose, stability_threshold, Q_ini )
%Q_LEARNING with epsilon-greedy target policy for episodic or non-episodic MDPs.
%   [ PI, Q, episodes_count ] = Q_learning( problem, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose, stability_threshold, Q_ini )
%   Finds optimal policy and optimal state-action value function for the
%   problem iterating over n_episodes episodes with epsilon-greedy policy
%   using a constant alpha as step-size sequence. 
%   If n_episodes = inf, then iterations will continue until Q changes less 
%   than stability_threshold between iterations. 
%   An episode is terminated if it reaches a terminal state or if the 
%   accumulated discount factor becomes smaller than discount_threshold. 
%   Discount threshold can't be 0 if the MPD is non-episodic. 
%   Greedy policies select all actions whose value is not worse than the 
%   best minus tolerance..

narginchk(8,9);

% Get parameters
n_states =          problem.n_states;
n_actions =         problem.n_actions;
gamma =             problem.gamma;
terminal_states =   problem.terminal_states;

% Initialize Q arbitrarily for all state-action pairs if no initial Q is
% provided
if nargin < 9
    Q = 20*rand(n_states,n_actions)-10;
else
    Q = Q_ini;
end

% Initialice Q to 0 for all terminal states
for ts = terminal_states
    Q(ts,:) = 0;
end

step = floor(n_episodes/100);
episodes_count = 0;
delta = inf;
while episodes_count < n_episodes && delta > stability_threshold
    episodes_count = episodes_count + 1;
    delta = 0;
    if verbose
        if n_episodes == inf
            disp(['Q-learning episode ',num2str(episodes_count)])
        elseif ~mod(episodes_count,step)
            disp(['Q-learning episode ',num2str(episodes_count),' of ',num2str(n_episodes)])
        end
    end
    
    % Initialize s
    s = problem.sampleInitialState();
    
    % Initialize loop variables
    discount = 1; % Accumulated discount
    is_terminal = problem.isTerminal(s);
    
    while discount > discount_threshold && ~is_terminal
        discount = discount*gamma; % Update accumulated discount
        
        % Choose action using e-greedy policy from current Q
        a = problem.sampleStateEpsilonGreedyPolicy(Q,tolerance,s,epsilon);
        % Take action a and observe r and s_next
        [s_next, r, is_terminal] = problem.sampleTransition(s,a);
        % Find greedy action for next state
        greedy_a_next = problem.sampleStateGreedyPolicy(Q,tolerance,s_next);
        % Save old Q
        Q_old = Q; 
        % Update Q(s,a)
        Q(s,a) = Q(s,a) + alpha*(r+gamma*Q(s_next, greedy_a_next)-Q(s,a));
        % Update s
        s = s_next;
        % Check stability
        delta = max(max(max(abs(Q_old-Q))), delta);
    end
end
% Calculate greedy policy
PI = problem.getGreedyPolicy(Q,tolerance);
end