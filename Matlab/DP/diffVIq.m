function [ PI, Q_diff, v_diff, n_it ] = diffVIq( problems, epsilon, verbose, neighbours )
%DIFFVIQ Diffusion Value Iteration for state-action value functions.
%   [ PI, Q, v, n_it ] = diffVIq( problems, epsilon, verbose, neighbours )
%
%   [ PI, Q, v, n_it ] = diffVIq( problems, epsilon, verbose )

narginchk(3,4);

% Get parameters
n_states = problems(1).n_states;
n_actions = problems(1).n_actions;
terminal_states = problems(1).terminal_states;
n_problems = length(problems);

% Set neighbours as fully connected if no matrix is provided
if nargin < 4
    neighbours = 1/(n_problems)*ones(n_problems);
end

% Initialize diffusion Q arbitrarily for all state-action pairs
Q = 20*rand(n_states,n_actions,n_problems)-10;

% Initialize diffusion Q to 0 for all terminal states
for s = terminal_states
    Q(s,:,:) = 0;
end

% Initialize local Q
Q_local = Q;

% Initialize loop variables
n_it = 0;
delta = inf;
while delta >= epsilon
    n_it = n_it+1;
    if verbose
        disp(['Diff. VI iteration ',num2str(n_it)])
    end
    
    Q_old = Q; % Save old Q
    
    for k = 1:n_problems % For each problem
        problem = problems(k);
        % Update local Q with Optimal Bellman Operator from diffusion Q
        Q_local(:,:,k) = bellmanOperatorQ(problem, Q_old(:,:,k));
    end
    
    % Diffusion
    for k = 1:n_problems % For each problem
        Q(:,:,k) = 0;
        for kk = 1:n_problems % For each possible neighbour
            % Update diffusion Qk with local Qkk
            Q(:,:,k) = Q(:,:,k) + Q_local(:,:,kk)* neighbours(k,kk);
        end
    end
    
    % Check stability
    delta = max(max(max(abs(Q_old-Q))));
end
% Get greedy policy
Q_diff = Q(:,:,1);
PI = problems(1).getGreedyPolicy(Q(:,:,1), epsilon);
v_diff = getVfromQ(PI,Q_diff);
end