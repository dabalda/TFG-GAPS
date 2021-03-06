function [ PI_diff, Q_diff, v_diff, episodes_count, n_samples ] = diffSARSA( problems, n_episodes, epsilon, alpha, discount_threshold, tolerance, verbose, stability_threshold, min_stable_steps, neighbours )
%DIFFSARSA

narginchk(9,10);

% Get parameters
n_states = problems(1).n_states;
n_actions = problems(1).n_actions;
n_problems = length(problems);

% Set neighbours if no matrix is provided
if nargin < 10
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

% Alpha setup
if alpha ~= 'decreasing'
    alpha_n = alpha; 
end

% Number of samples for each state-action pair
n_samples = zeros(n_states, n_actions, n_problems);

% Initialize loop variables
discount = ones(n_problems,1); % Accumulated discount
is_terminal = ones(n_problems,1);
episodes_count = zeros(n_problems,1);
stable_steps = 0;
s = zeros(n_problems,1);
a = zeros(n_problems,1);

while mean(episodes_count) < n_episodes && stable_steps < min_stable_steps
    
    Q_old = Q;
    for k = 1:n_problems % Local steps
        
        problem = problems(k);
        gamma = problem.gamma;
        
        if is_terminal(k) % Initialize new episode
            % Initialize to a non terminal state
            while is_terminal(k)
                s(k) = problem.sampleInitialState();
                is_terminal(k) = problem.isTerminal(s(k));
            end
            % Choose action using e-greedy policy from current Q
            a(k) = problem.sampleStateEpsilonGreedyPolicy(Q(:,:,k),tolerance,s(k),epsilon);
            
            % Initialize loop variables
            discount(k) = 1; % Accumulated discount
            
        end % New step
        
        discount(k) = discount(k)*gamma; % Update accumulated discount
        
        % Take action a and observe r and s_next
        [s_next, r, is_terminal(k)] = sampleTransition(problem, s(k), a(k));
        % Choose action using e-greedy policy from current Q
        a_next = problem.sampleStateEpsilonGreedyPolicy(Q(:,:,k),tolerance,s_next,epsilon);
        % Update sample count
        n_samples(s(k),a(k),k) = n_samples(s(k),a(k),k) + 1;
        % If alpha is decreasing, set alpha_n
        if alpha == 'decreasing'
            alpha_n = 1/(n_samples(s(k),a(k),k)^(sqrt(0.5)));
        end
        % Update Q(s,a)
        Q_local(s(k),a(k),k) = Q(s(k),a(k),k) + alpha_n*(r+gamma*Q(s_next,a_next,k)-Q(s(k),a(k),k));
        % Update s & a
        s(k) = s_next;
        a(k) = a_next;
        
        % Force termination after a number of steps
        is_terminal(k) = is_terminal(k) || discount(k) < discount_threshold;
        
        if is_terminal(k)
            episodes_count(k) = episodes_count(k)+1;
            if verbose && episodes_count(k) <= n_episodes && episodes_count(k) > 0
                format_p = '%'+num2str( length(num2str(n_problems)))+'u';
                p = num2str(k, format_p);
                disp(['Diff. SARSA problem ',p,' / ',num2str(n_problems),...
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
    % Check stability
    delta = sum(sum(sum(abs(Q_old-Q))))/sum(sum(sum(abs(Q_old))));
    if delta < stability_threshold
        stable_steps = stable_steps + 1;
        if verbose >= 2
            disp(['Diff. SARSA, ',num2str(stable_steps), ' stable steps out of ',num2str(min_stable_steps)]);
        end
    else
        stable_steps = 0;
    end
    
end % of all episodes

% Get greedy policy
Q_diff = Q(:,:,1);
PI_diff = problems(1).getGreedyPolicy(Q(:,:,1), epsilon);
v_diff = getVfromQ(PI_diff,Q_diff);
end

