classdef (Abstract) Problem < handle
    %PROBLEM Extend this abstract class in order to define MDPs.
    %   Includes common properties and methods.
    
    properties
        n_states;           % Number of states
        n_actions;          % Number of actions
        gamma;              % Discount factor
        Pssa;               % Transition probability matrix
        Rssa;               % Reward matrix
        initial_states;     % Initial states probability vector
        terminal_states;    % List of terminal states
    end
    
    methods
        
        % Constructor
        
        function obj = Problem(parameters)
            % Constructor.
            % Accepts as input a struct which has to contain these fields:
            %   'gamma': discount factor.
            %   Other fields required by the extended class.
            if ~isempty(parameters)
                obj.setNStates(parameters);
                obj.setNActions(parameters);
                obj.setGamma(parameters);
                obj.setPssa(parameters);
                obj.setRssa(parameters);
                obj.setInitialStates(parameters);
                obj.setTerminalStates(parameters);
            end
        end
        
        % States and rewards
        
        function is_terminal = isTerminal(obj, state)
            % isTerminal Returns true if state is a terminal state.
            
            is_terminal = any(obj.terminal_states == state);
        end
        
        function s = sampleInitialState(obj)
            % sampleInitialState Returns index of an initial state sampled
            % from initial_states distribution.
            
            s = discretesample(obj.initial_states,1);
        end
        
        function [sf, r, is_terminal] = sampleTransition(obj, si, a)
            % sampleTransition Returns new state, reward and whether new
            % state is terminal, given an initial state and an action.
            sf = discretesample(obj.Pssa(si,:,a),1);
            r = obj.Rssa(si,sf,a);
            is_terminal = any(obj.terminal_states == sf);
        end
        
        % Policies
        function PI = getRandomPolicy(obj)
            % getRandomPolicy Returns the stochastic random policy for this
            % problem, which gives equal probability to all possible
            % actions.
            ns = obj.n_states;
            na = obj.n_actions;
            PI = (1/na)*ones(ns,na);
        end
        
        function PI = sampleRandomPolicy(obj)
            % sampleRandomPolicy Returns a random deterministic policy for 
            % this problem, sampled from the stochastic random policy
            ns = obj.n_states;
            na = obj.n_actions;
            PI = getRandomPolicy(obj);
            for s = 1:ns
                a = discretesample(PI(s,:),1);
                PI(s,:) = zeros(na,1);
                PI(s,a) = 1;
            end        
        end
        
        function PIsa = getGreedyPolicy(obj, Q, tolerance)
            % getGreedyPolicy Returns greedy policy matrix given a
            % state-action value vector and a tolerance for equal values.
            
            ns = obj.n_states;
            na = obj.n_actions;
            % Initialize policy probability matrix
            %             PIsa = zeros(ns,ns*na);
            %             for s = 1:obj.n_states
            %                 PIsa(s,(s-1)*na+1:s*na) = getStateGreedyPolicy(obj,Q,tolerance,s);
            %             end
            PIsa = zeros(ns,na);
            for s = 1:obj.n_states
                PIsa(s,:) = getStateGreedyPolicy(obj,Q,tolerance,s);
            end
        end
        
        function PIa = getStateGreedyPolicy(obj, Q, tolerance, s)
            % getStateGreedyPolicy Returns greedy policy vector for state
            % s, given a state-action value vector and a tolerance for
            % equal values.
            
            % Find greedy action for chosen state
            action_greedy = find(Q(s,:) >= max(Q(s,:))- tolerance);
            % Initialize policy probability vector
            PIa = zeros(obj.n_actions,1);
            % Assign probability to chosen actions
            for a = 1:obj.n_actions
                if any(action_greedy == a)
                    PIa(a) = 1/length(action_greedy);
                end
            end
        end
        
        function a = sampleStateGreedyPolicy(obj, Q, tolerance, s)
            % sampleStateGreedyPolicy Returns a greedy action for state s
            % sampled from all greedy actions for this state.
            
            % Get policy
            PIa = getStateGreedyPolicy(obj,Q,tolerance,s);
            % Sample policy
            a = discretesample(PIa,1);
        end
        
        function a = sampleStateEpsilonGreedyPolicy(obj, Q, tolerance, s, epsilon)
            % sampleStateEpsilonGreedyPolicy Returns an action for state s
            % sampling from an epsilon-greedy policy.
            
            % Choose between greedy or random
            greedy = rand() >= epsilon;
            
            if greedy
                a = sampleStateGreedyPolicy(obj,Q,tolerance,s);
            else
                % Choose an action randomly
                a = randi(obj.n_actions);
            end
        end
    end
    
    methods (Abstract)
        
        setNStates(obj, parameters);
        setNActions(obj, parameters);
        setGamma(obj, parameters);
        setPssa(obj, parameters);
        setRssa(obj, parameters);
        setInitialStates(obj, parameters);
        setTerminalStates(obj, parameters);
    end
    
end