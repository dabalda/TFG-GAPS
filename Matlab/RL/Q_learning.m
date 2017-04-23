function [ PI, Q ] = Q_learning( problem, n_episodes, epsilon, alpha, discount_threshold, tolerance )
%Q_LEARNING with epsilon-greedy target policy for episodic or non-episodic MDPs.
%   [ PI, Q ] = Q_learning(problem,n_episodes,epsilon,alpha,discount_threshold,tolerance)
%   Finds optimal policy and optimal state-action value function for the
%   problem iterating over n_episodes episodes with epsilon-greedy policy
%   using a constant alpha as step-size sequence. An episode is terminated
%   if it reaches a terminal state or if the accumulated discount factor
%   becomes smaller than discount_threshold. Discount threshold can't be 0
%   if the MPD is non-episodic. Greedy policies select all actions whose
%   value is not worse than the best minus tolerance.

% Get parameters
n_states = problem.n_states;
n_actions = problem.n_actions;
gamma = problem.gamma;
terminal_states = problem.terminal_states;

% Initialize Q arbitrarily for all state-action pairs
Q = 20*rand(n_states,n_actions)-10;

% Initialice Q to 0 for all terminal states
for s = terminal_states
    Q(s,:) = 0;
end

for i = 1:n_episodes
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
        greedy_a_next = problem.sampleStateEpsilonGreedyPolicy(Q,tolerance,s_next,epsilon);
        % Update Q(s,a)
        Q(s,a) = Q(s,a) + alpha*(r+gamma*Q(s_next, greedy_a_next)-Q(s,a));
        % Update s
        s = s_next;
    end
end
% Calculate greedy policy
PI = problem.getGreedyPolicy(Q,tolerance);
end