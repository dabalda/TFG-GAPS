classdef (Abstract) Problem < handle
    %PROBLEM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_states;
        n_actions;
        gamma;
        Pssa;
        Rssa;
        initial_states;
        terminal_states;
    end
    
    methods
        function obj = Problem(parameters) % Constructor
            obj.setNStates(parameters);
            obj.setNActions(parameters);
            obj.setGamma(parameters);
            obj.setPssa(parameters);
            obj.setRssa(parameters);
            obj.setInitialStates(parameters);
            obj.setTerminalStates(parameters);
        end
        
        function new_state = getNewState(obj, old_state, action)
            % Choose row of transition probabilities corresponding to
            % old_state and action
            p = obj.Pssa(old_state,:,action);
            % Calculate cumulative distribution function
            cdf_p = cumsum([0,p]);
            % Choose new state
            new_state = sum(rand >= cdf_p);
        end
        
        function reward = getReward(obj, old_state, new_state, action)
            % Find reward
            reward = obj.Rssa(old_state, new_state, action);
        end
        
        function is_terminal = isTerminal(obj,state)
            is_terminal = any(obj.terminal_states == state)
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

