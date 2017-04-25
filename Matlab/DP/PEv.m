function [ v ] = PEv( problem, PI, epsilon )
%PEV Policy Evaluation for state-value functions.
%   [ v ] = PEv( problem, PI, epsilon )
%   Evaluates state value function for policy PI. Loop ends when delta <
%   epsilon.

% Get parameters
n_states =  problem.n_states;
n_actions = problem.n_actions;
gamma =     problem.gamma;
P =         problem.Pssa;
R =         problem.Rssa;

% Initialize v(s) arbitrarily
v = 20*rand(n_states,1)-10;

% Initialize loop
delta = inf;

while delta >= epsilon
    v_old = v;
    for si = 1:n_states % For each initial state
        v(si) = 0;
        for sf = 1:n_states % For each next state
            for a = 1:n_actions % For each action
                v(si) = v(si) + PI(si,a)*(R(si,sf,a)+gamma*P(si,sf,a)*v_old(sf));
            end
        end
    end
    delta = norm(v_old-v,inf);
end


end

