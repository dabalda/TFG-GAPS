function [ PI, Q, episodes_count, n_samples, G, Q_hist ] = Q_learning_for_benchmarking_v3( problem, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose, Q_ini )
%Q_LEARNING_for_benchmarking_v3 with epsilon-greedy behavior policy for episodic or non-episodic MDPs.
%   [ PI, Q, episodes_count, n_samples, , Q_hist ] = Q_learning_for_benchmarking_v3( problem, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose, Q_ini )
%   Finds optimal policy and optimal state-action value function for the
%   problem iterating over n_episodes episodes with epsilon-greedy policy
%   using a constant or decreasing alpha as step-size sequence. 
%   An episode is terminated if it reaches a terminal state or if the 
%   accumulated discount factor becomes smaller than discount_threshold. 
%   Discount threshold can't be 0 if the MPD is non-episodic. 
%   Greedy policies select all actions whose value is not worse than the 
%   best minus tolerance.

narginchk(7,8);

% Get parameters
n_states =          problem.n_states;
n_actions =         problem.n_actions;
gamma =             problem.gamma;
terminal_states =   problem.terminal_states;

% Initialize Q arbitrarily for all state-action pairs if no initial Q is
% provided
if nargin < 8
    Q = 20*rand(n_states,n_actions)-10;
else
    Q = Q_ini;
end

% Initialice Q to 0 for all terminal states
for ts = terminal_states
    Q(ts,:) = 0;
end

% Q history
Q_hist = zeros(n_states, n_actions, n_episodes);

% Alpha setup
if alpha ~= 'decreasing'
    alpha_n = alpha; 
end

% Number of samples for each state-action pair
n_samples = zeros(n_states, n_actions);

% Instant cummulative reward for each episode
G = zeros(n_episodes, 1);

step = floor(n_episodes/100);
episodes_count = 0;
while episodes_count < n_episodes
    episodes_count = episodes_count + 1;
    
    if verbose && ~mod(episodes_count,step)
        disp(['Q-learning episode ',num2str(episodes_count),' of ',num2str(n_episodes)])
    end
    
    % Initialize s
    s = problem.sampleInitialState();
    
    % Initialize loop variables
    discount = 1; % Accumulated discount
    is_terminal = problem.isTerminal(s);
    % Number of steps taken in current episode
    n_steps = 0;
    
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
        Q_hist(:,:,episodes_count) = Q;
        Q(s,a) = Q(s,a) + alpha_n*(r+gamma*Q(s_next, greedy_a_next)-Q(s,a));
        G(episodes_count) = G(episodes_count) + r*gamma^n_steps;
        n_steps = n_steps + 1;
        % Update s
        s = s_next;       
    end
end
% Calculate greedy policy
PI = problem.getGreedyPolicy(Q,tolerance);
end