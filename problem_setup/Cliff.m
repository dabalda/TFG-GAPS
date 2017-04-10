classdef Cliff < Problem
    %RANDOMWALK Summary of this class goes here
    %   Detailed explanation goes here
    
    methods
        function obj = Cliff(parameters) % Constructor
            % Call superclass constructor with parameters
            obj@Problem(parameters);
        end
        
        function setNStates(obj,~)
            obj.n_states = 48; % 4x12 grid
        end
        
        function setNActions(obj,~)
            obj.n_actions = 4; % North, East, South, West
        end
        
        function setGamma(obj, parameters)
            obj.gamma = parameters.gamma;
        end
        
        function setPssa(obj, parameters)
            e = parameters.slope_east;
            n = parameters.slope_north;
            ns = obj.n_states;
            na = obj.n_actions;
            Pssa = zeros(ns, ns, na);
            %TODO
            % Calculate transition probabilities
            
            obj.Pssa = Pssa;
        end
        
        function setRssa(obj,~)
            ns = obj.n_states;
            na = obj.n_actions;
            Rssa = -1*ones(ns,ns,na); % Reward is -1 for most transitions
            Rssa(:,2:11,:) = -100; % Reward is -100 when falling
            Rssa(:,ns,:) = 0; % Reward is 0 when reaching the terminal state
            obj.Rssa = Rssa;
        end
        
        function setInitialStates(obj,~)
            ns = obj.n_states;
            is = zeros(ns,1);
            is(1) = 1; % Initial state is state 1 with probability 1
            obj.initial_states = is;
        end
        
        function setTerminalStates(obj,~)
            ns = obj.n_states;
            te = [12]; % Terminal state is state 12 (southeastern corner)
            obj.terminal_states = te;
        end
    end
    
end

