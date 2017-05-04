function [ PI, Q_diff, v_diff, episodes_count ] = diffQ_learning( problems, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose, neighbours )
%DIFFQ_LEARNING

narginchk(7,8);

% Get parameters
n_states = problems(1).n_states;
n_actions = problems(1).n_actions;
n_problems = length(problems);

% Set neighbours if no matrix is provided
if nargin < 8
    neighbours = 1/(n_problems)*ones(n_problems);
end

% Initialize diffusion Q arbitrarily for all state-action pairs
Q = 20*rand(n_states,n_actions,n_problems)-10;

% Initialize diffusion Q to 0 for all terminal states
for k = 1:n_problems
    terminal_states = problems(k).terminal_states;
    for ts = terminal_states
        Q(ts,:,k) = 0;
    end
end

% Initialize local Q
Q_local = Q;

% Initialize loop variables
discount = ones(n_problems,1); % Accumulated discount
is_terminal = ones(n_problems,1);
episodes_count = zeros(n_problems,1);
s = zeros(n_problems,1);

while min(episodes_count) < n_episodes
    
    for k = 1:n_problems % Local steps
        
        problem = problems(k);
        gamma = problem.gamma;
        
        if is_terminal(k) % Initialize new episode
            % Initialize to a non terminal state
            while is_terminal(k)
                s(k) = problem.sampleInitialState();
                is_terminal(k) = problem.isTerminal(s(k));
            end

            % Initialize loop variables
            discount(k) = 1; % Accumulated discount
            
        end % New step
        
        discount(k) = discount(k)*gamma; % Update accumulated discount
        
        % Choose action using e-greedy policy from current Q
        a = problem.sampleStateEpsilonGreedyPolicy(Q(:,:,k),tolerance,s(k),epsilon);
        % Take action a and observe r and s_next
        [s_next, r, is_terminal(k)] = sampleTransition(problem, s(k), a);
        % Find greedy action for next state
        greedy_a_next = problem.sampleStateGreedyPolicy(Q(:,:,k),tolerance,s_next);               
        % Update Q(s,a)
        Q_local(s(k),a,k) = Q(s(k),a,k) + alpha*(r+gamma*Q(s_next,greedy_a_next,k)-Q(s(k),a,k));
        % Update s
        s(k) = s_next;
        
        % Force termination after a number of steps
        is_terminal(k) = is_terminal(k) || discount(k) < discount_threshold;
        
        if is_terminal(k)
            episodes_count(k) = episodes_count(k)+1;
            if verbose && episodes_count(k) <= n_episodes && episodes_count(k) > 0
                format_p = '%'+num2str( length(num2str(n_problems)))+'u';
                p = num2str(k, format_p);
                disp(['Diff. Q-learning problem ',p,' / ',num2str(n_problems),...
                    ', episode ',num2str(episodes_count(k)),' / ',num2str(n_episodes),' done.'])
            end
        end
        
    end % of local steps
    
    % Diffusion
    for k = 1:n_problems % For each problem
        Q(:,:,k) = 0;
        for kk = 1:n_problems % For each possible neighbour
            % Update diffusion Qk with local Qkk
            Q(:,:,k) = Q(:,:,k) + Q_local(:,:,kk)* neighbours(k,kk);
        end
    end
    
end % of all episodes

% Get greedy policy
Q_diff = Q(:,:,1);
PI = problems(1).getGreedyPolicy(Q(:,:,1), epsilon);
v_diff = getVfromQ(PI,Q_diff);
end
