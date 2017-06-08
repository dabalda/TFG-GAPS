classdef RandomWalk < Problem
    %RANDOMWALK Random Walk problem definition
    %   Parameters:
    %   gamma
    %   prob_desired_right
    %   prob_desired_left
    %   all_states_initial
    
    methods
        function obj = RandomWalk(parameters)
            % Constructor
            
            % Call superclass constructor with parameters
            if nargin == 0
                parameters = struct([]);
            end
            obj@Problem(parameters);
        end
        
        function setNStates(obj,~)
            obj.n_states = 7; % Leftmost state is number 1, rightmost is 7
        end
        
        function setNActions(obj,~)
            obj.n_actions = 2; % Left, Right
        end
        
        function setGamma(obj, parameters)
            obj.gamma = parameters.gamma;
        end
        
        function setPssa(obj, parameters)
            pl = parameters.prob_desired_left;
            pr = parameters.prob_desired_right;
            ns = obj.n_states;
            na = obj.n_actions;
            Pssa = zeros(ns, ns, na);
            
            for si = 1:ns
                for sf = 1:ns
                    for a = 1:na
                        if (si == 1 || si == ns) % End states
                            if si == sf
                                Pssa(si,sf,a) = 1;
                            end
                        elseif sf == si-1 % Moving left
                            if a == 1 % When trying to move left
                                Pssa(si,sf,a) = pl;
                            elseif a == 2 % When trying to move right
                                Pssa(si,sf,a) = 1-pr;
                            end
                        elseif sf == si+1 % Moving right
                            if a == 1 % When trying to move left
                                Pssa(si,sf,a) = 1-pl;
                            elseif a == 2 % When trying to move right
                                Pssa(si,sf,a) = pr;
                            end
                        end
                    end
                end
            end
            obj.Pssa = Pssa;
        end
        
        function setRssa(obj,~)
            ns = obj.n_states;
            na = obj.n_actions;
            Rssa = zeros(ns,ns,na);
            Rssa(:,ns,:) = 1;
            Rssa(ns,ns,:) = 0;
            obj.Rssa = Rssa;
        end
        
        function setInitialStates(obj, parameters)
            ns = obj.n_states;
            is = zeros(ns,1);
            if parameters.all_states_initial
                is(2:ns-1) = 1/(ns-2);
            else              
                is(ceil(ns/2)) = 1;
            end
            obj.initial_states = is;
        end
        
        function setTerminalStates(obj,~)
            ns = obj.n_states;
            ts = [1,ns];
            obj.terminal_states = ts;
        end
        
        function plotPssa(obj)
            figure
            for a = 1:obj.n_actions
                subplot(2,1,a)
                action_name = {'Left','Right'};
                title(action_name{a});
                hold
                for si = 1:obj.n_states
                    for sf =  1:obj.n_states
                        if obj.Pssa(si,sf,a) > 0
                            q = quiver(si,0,(sf-si)*obj.Pssa(si,sf,a),0);
                            q.Color = 'red';
                            q.LineWidth = 2;
                            % q.ShowArrowHead = 'off';
                            % q.MaxHeadSize = 1;
                            q.MaxHeadSize = 1/(obj.Pssa(si,sf,a));
                            
                            q.Marker = 'o';
                            q.MarkerFaceColor = 'blue';
                            q.MarkerEdgeColor = 'blue';
                            
                        end
                    end
                end
                axis off
            end
        end
    end
    
end