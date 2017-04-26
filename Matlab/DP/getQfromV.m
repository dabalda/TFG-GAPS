function [ Q ] = getQfromV( problem, v )
%GETQFROMV Get Q(s,a) from v(s).
%   [ Q ] = getQfromV( problem, v )

% Get parameters
n_states =  problem.n_states;
n_actions = problem.n_actions;
gamma =     problem.gamma;
P =         problem.Pssa;
R =         problem.Rssa;

% Repeat v for each initial state and action
v_ssa = repmat(v(:)',[n_states,1,n_actions]);
% Partial result
v_2 = R+gamma.*v_ssa;
% Get Q
Q = squeeze(sum(P.*v_2, 2));
end

