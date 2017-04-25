function [ PI, q_opt ] = diff_SARSA( problems, n_episodes, epsilon, alpha, discount_threshold, tolerance, taus )
%DIFF_SARSA

n_states = problems(1).n_states;
n_actions = problems(1).n_actions;
terminal_states = problems(1).terminal_states;
n_problems = length(problems);

% Initialize diffusion Q arbitrarily for all state-action pairs
Q = 20*rand(n_states,n_actions,n_problems)-10;

% Initialize diffusion Q to 0 for all terminal states
for s = terminal_states
    Q(s,:) = 0;
end

% Initialize local Q 
Q_local = Q;

for i = 1:n_episodes
    
    
    
    for j = 1:n_problems
        disp(['Diff. SARSA episode ',num2str(i),' / ',num2str(n_episodes),...
            ', problem ',num2str(j),' / ',num2str(n_problems)])
       
        problem = problems(j);
        
        gamma = problem.gamma;
        terminal_states = problem.terminal_states;
        
        % Initialize s
        s = randi([1, n_states]);
        % Choose action using e-greedy policy from current q
        a = e_greedy(problem, q_local(:,j), s, epsilon);
        
        discount = 1; % Total discount
        while discount > discount_threshold
            discount = discount*gamma;
            
            % Take action a and observe r and s_next
            [s_next, r] = action_effect(problem, s, a);
            % Choose action using e-greedy policy from current q
            a_next = e_greedy(problem, q_local(:,j), s_next, epsilon);
            
            currentSA = (s-1)*n_actions+a;
            nextSA = (s_next-1)*n_actions+a_next;
            
            % Update q(s,a)
            q_local(currentSA,j) = q_local(currentSA,j) + alpha*(r+gamma*Q(nextSA,j)-q_local(currentSA,j));
            
            % Update s & a
            s = s_next;
            a = a_next;
            % if s is terminal state: break
            if any(terminal_states==s)
                break
            end
        end
        
    end
    
    q = q_local*taus';
    
end
% Calculate greedy policy
q_opt = q(:,1);
PI = greedy_policy(problems(1), q_opt, tolerance);
end

