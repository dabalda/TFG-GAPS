function [ Q ] = PEq( problem, PI, epsilon )
%PEV Policy Evaluation for state-value functions.
%   [ Q ] = PEv( problem, PI, epsilon )
%   Evaluates state-action value function for policy PI. Loop ends when
%   delta < epsilon.

% Get parameters
n_states =  problem.n_states;
n_actions = problem.n_actions;
gamma =     problem.gamma;
P =         problem.Pssa;
R =         problem.Rssa;

% Initialize v(s) arbitrarily
Q = 20*rand(n_states,n_actions)-10;

% Initialize loop
delta = inf;

while delta >= epsilon
    Q_old = Q;
    for si = 1:n_states % For each initial state
        for ai = 1:n_actions % For each initial action
            Q(si,ai) = 0;
            for sf = 1:n_states % For each next state
                for af = 1:n_actions % For each next action
                    Q(si,ai) = Q(si,ai) + R(si,sf,ai)+gamma*P(si,sf,ai)*PI(sf,af)*Q_old(sf,af);
                end
            end
        end
       
    end
    delta = max(max(abs(Q_old-Q)));
end


end

