function [ PI, v, n_it ] = PIv( problem, epsilon, PI_ini )
%PIV Policy Iteration for state value functions.
%   [ PI, v, Q, n_it ] = PIv( problem, epsilon, PI_ini ) finds optimal
%   policy PI and state value vector for the problem. Policy evaluation
%   loop ends when delta < epsilon. The initial policy is PI_ini or the
%   random policy if PI_ini = [].

% Initialize PI to random policy if none is supplied
PI = PI_ini;
if isequal(PI, [])
    PI = random_det_policy(problem);
end
policy_stable = false;
n = 0;

while ~policy_stable % Policy iteration main loop
    n = n+1;
    disp(['Policy Iteration ',num2str(n)])
    % Policy evaluation
    [v_pi, ~] = PEv(problem, PI, epsilon);
    policy_stable = true;
    
    % Policy improvement
    PI_temp = PI;
end

