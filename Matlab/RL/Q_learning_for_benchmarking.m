function [ PI, Q, episodes_count, n_samples ] = Q_learning_for_benchmarking( problem, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose, stability_threshold, min_stable_ep, Q_opt, Q_ini)
%Q_LEARNING_FOR_BENCHMARKING with epsilon-greedy target policy for episodic or non-episodic MDPs.
%   [ PI, Q, episodes_count ] = Q_learning( problem, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose, stability_threshold, Q_ini )
%   Finds optimal policy and optimal state-action value function for the
%   problem iterating over n_episodes episodes with epsilon-greedy policy
%   using a constant or decreasing alpha as step-size sequence. 
%   If n_episodes = inf, then iterations will continue until Q is closer 
%   than stability_threshold to the optimal Q.
%   An episode is terminated if it reaches a terminal state or if the 
%   accumulated discount factor becomes smaller than discount_threshold. 
%   Discount threshold can't be 0 if the MPD is non-episodic. 
%   Greedy policies select all actions whose value is not worse than the 
%   best minus tolerance.

narginchk(10,11);

% Get parameters
n_states =          problem.n_states;
n_actions =         problem.n_actions;
gamma =             problem.gamma;
terminal_states =   problem.terminal_states;

% Initialize Q arbitrarily for all state-action pairs if no initial Q is
% provided
if nargin < 11
    Q = 20*rand(n_states,n_actions)-10;
else
    Q = Q_ini;
end

% Initialice Q to 0 for all terminal states
for ts = terminal_states
    Q(ts,:) = 0;
end

% Alpha setup
if alpha ~= 'decreasing'
    alpha_n = alpha; 
end

% Number of samples for each state-action pair
n_samples = zeros(n_states, n_actions);

step = floor(n_episodes/100);
episodes_count = 0;
stable_ep = 0;
while episodes_count < n_episodes && stable_ep < min_stable_ep%delta >= stability_threshold
    episodes_count = episodes_count + 1;
    
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
        % Update sample count
        n_samples(s,a) = n_samples(s,a) + 1;
        % If alpha is decreasing, set alpha_n
        if alpha == 'decreasing'
            alpha_n = 1/(n_samples(s,a)^(sqrt(0.5)));
        end
        % Update Q(s,a)
        Q(s,a) = Q(s,a) + alpha_n*(r+gamma*Q(s_next, greedy_a_next)-Q(s,a));
        % Update s
        s = s_next;       
    end
    % Check similarity to optimal Q
    delta = sum(sum(abs(Q_opt-Q)))/sum(sum(abs(Q_opt)));
    if delta < stability_threshold
        stable_ep = stable_ep + 1;
        if verbose >= 2
            disp(['Q-learning, ',num2str(stable_ep), ' stable episodes out of ',num2str(min_stable_ep)]);
        end
    else
        stable_ep = 0;
    end
end
% Calculate greedy policy
PI = problem.getGreedyPolicy(Q,tolerance);
end