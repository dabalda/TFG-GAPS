function [ PI, Q ] = SARSA( problem, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose )
%SARSA with epsilon-greedy target policy for episodic or non-episodic MDPs.
%   [ PI, Q ] = SARSA(problem,n_episodes,epsilon,alpha,discount_threshold,tolerance, verbose)
%   Finds optimal policy and optimal state-action value function for the
%   problem iterating over n_episodes episodes with epsilon-greedy policy
%   using a constant alpha as step-size sequence. An episode is terminated
%   if it reaches a terminal state or if the accumulated discount factor
%   becomes smaller than discount_threshold. Discount threshold can't be 0
%   if the MPD is non-episodic. Greedy policies select all actions whose
%   value is not worse than the best minus tolerance.

% Get parameters
n_states =          problem.n_states;
n_actions =         problem.n_actions;
gamma =             problem.gamma;
terminal_states =   problem.terminal_states;

% Initialize Q arbitrarily for all state-action pairs
Q = 20*rand(n_states,n_actions)-10;

% Initialize Q to 0 for all terminal states
for ts = terminal_states
    Q(ts,:) = 0;
end

step = floor(n_episodes/100);
for i = 1:n_episodes
    if ~mod(i,step) && verbose
        disp(['SARSA episode ',num2str(i),' of ',num2str(n_episodes)])
    end    
    
    % Initialize s
    s = problem.sampleInitialState();
    % Choose action using e-greedy policy from current Q
    a = problem.sampleStateEpsilonGreedyPolicy(Q,tolerance,s,epsilon);
    
    % Initialize loop variables
    discount = 1; % Accumulated discount
    is_terminal = problem.isTerminal(s);
    
    while discount > discount_threshold && ~is_terminal
        discount = discount*gamma; % Update accumulated discount
        
        % Take action a and observe r and s_next
        [s_next, r, is_terminal] = sampleTransition(problem, s, a);
        % Choose action using e-greedy policy from current Q
        a_next = problem.sampleStateEpsilonGreedyPolicy(Q,tolerance,s_next,epsilon);
        % Update Q(s,a)
        Q(s,a) = Q(s,a) + alpha*(r+gamma*Q(s_next,a_next)-Q(s,a));
        % Update s & a
        s = s_next;
        a = a_next;
    end
end
% Get greedy policy
PI = problem.getGreedyPolicy(Q,tolerance);
end

