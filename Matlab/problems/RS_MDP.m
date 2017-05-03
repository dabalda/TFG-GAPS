classdef RS_MDP < Problem
    %RS_MDP Summary of this class goes here
    %   Policy Evaluation with Temporal Differences: A Survey and Comparison Christoph Dann
    %   p. 851
    %
    % Parameters:
    % gamma: discount factor
    % n_states: Number of states
    % n_actions: Number of actions
    
    methods
        function obj = RS_MDP(parameters)
            % Constructor
            
            % Call superclass constructor with parameters
            if nargin == 0
                parameters = struct([]);
            end
            obj@Problem(parameters);
        end
        
        function setNStates(obj, parameters)
            obj.n_states = parameters.n_states;
        end
        
        function setNActions(obj, parameters)
            obj.n_actions = parameters.n_actions;
        end
        
        function setGamma(obj, parameters)
            obj.gamma = parameters.gamma;
        end
        
        function setPssa(obj,~)
            ns = obj.n_states;
            na = obj.n_actions;
            % Generate normal distribution for P
            Pssa = rand(ns, ns, na);
            % Add a small constant to ensure ergodicity of the MDP
            Pssa = 1e-5 + Pssa;
            % Normalize P rows so that each sums 1                     
            for s = 1:ns
                for a = 1:na
                    Pssa(s,:,a) = Pssa(s,:,a)./sum(Pssa(s,:,a));
                end
            end
            obj.Pssa = Pssa;
        end
        
        function setRssa(obj,~)
            ns = obj.n_states;
            na = obj.n_actions;
            Rssa = rand(ns,ns,na);
            obj.Rssa = Rssa;
        end
        
        function setInitialStates(obj,~)
            ns = obj.n_states;
            is = (1/ns)*ones(ns,1); % All states

            obj.initial_states = is;
        end
        
        function setTerminalStates(obj,~)
            ts = []; % No terminal states
            obj.terminal_states = ts;
        end
    end
end
