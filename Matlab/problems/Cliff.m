classdef Cliff < Problem
    %RANDOMWALK Cliff Walking problem definition
    %   Parameters:
    %   gamma
    %   slope_east
    %   slope_north
    
    properties
        slopes;
    end
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
            
            % Calculate slopes as viewed from every direction
            rot90 = [0 -1; 1, 0];
            se = parameters.slope_east;
            sn = parameters.slope_north;
            
            obj.slopes = zeros(2,4);
            obj.slopes(:,1) = [se,sn];
            
            for i = 2:4
                obj.slopes(:,i) = rot90*obj.slopes(:,i-1);
            end
            
            ns = obj.n_states;
            na = obj.n_actions;
            Pssa = zeros(ns, ns, na);
            
            for si = 1:ns
                for a = 1:na
                    Pssa(si,:,a) = obj.getPssa(si,a);
                end
            end
            
            obj.Pssa = Pssa;
        end
        
        function setRssa(obj,~)
            ns = obj.n_states;
            na = obj.n_actions;
            Rssa = -1*ones(ns,ns,na); % Reward is -1 for most transitions
            Rssa(:,2:11,:) = -100; % Reward is -100 when falling
            Rssa(:,12,:) = 0; % Reward is 0 when reaching terminal state
            obj.Rssa = Rssa;
        end
        
        function setInitialStates(obj,~)
            ns = obj.n_states;
            is = zeros(ns,1);
            is(1) = 1; % Initial state is state 1 with probability 1
            obj.initial_states = is;
        end
        
        function setTerminalStates(obj,~)
            te = 12; % Terminal state is state 12 (southeastern corner)
            obj.terminal_states = te;
        end
        
        function [x, y] = getCoordinates(obj, state)
            x = mod(state-1, 12);
            y = ceil(state/12)-1;
        end
        
        function plotPssa(obj)
            figure
            for a = 1:obj.n_actions
                subplot(2,2,a)
                action_name = {'North','East','South','West'};
                title(action_name{a});
                hold
                for si = 1:obj.n_states
                    [xi, yi] = getCoordinates(obj, si);
                    for sf =  1:obj.n_states
                        if obj.Pssa(si,sf,a) > 0
                            [xf, yf] = getCoordinates(obj, sf);
                            q = quiver(xi,yi,(xf-xi)*obj.Pssa(si,sf,a),(yf-yi)*obj.Pssa(si,sf,a));
                            q.Color = 'red';
                            q.LineWidth = 2;
                            % q.ShowArrowHead = 'off';
                            % q.MaxHeadSize = 1;
                            
                            q.Marker = 'o';
                            q.MarkerFaceColor = 'blue';
                            q.MarkerEdgeColor = 'blue';
                        end
                    end
                end
            end
        end
        function plotPolicy(obj, PI)
            figure
            hold
            for si = 1:obj.n_states
                [xi, yi] = getCoordinates(obj, si);
                action_coord = [0,1;1,0;0,-1;-1,0];
                for a = 1:obj.n_actions
                    if PI(si,a) > 0
                        xf = xi + action_coord(a,1);
                        yf = yi + action_coord(a,2);
                        q = quiver(xi,yi,(xf-xi)*PI(si,a),(yf-yi)*PI(si,a));
                        q.Color = 'red';
                        q.LineWidth = 2;
                        % q.ShowArrowHead = 'off';
                        % q.MaxHeadSize = 1;
                        
                        q.Marker = 'o';
                        q.MarkerFaceColor = 'blue';
                        q.MarkerEdgeColor = 'blue';
                    end
                end
            end
            
        end
    end
    
    methods (Access = 'protected')
        function Pssa_simple = getSimplePssa(obj, a)
            slopeT = obj.slopes(:,a);
            if slopeT(2) >= 0
                perp1 = 1/2*(1-slopeT(2));
                perp2 = perp1*slopeT(1);
                PTemp = [1-abs(perp2), max(0, perp2), 0, max(0,-perp2)];
            else
                perp = 1/2*slopeT(1);
                paral = 1-abs(perp);
                forward = paral*(1+1/2*slopeT(2));
                PTemp = [forward, max(0, perp), 1-forward-abs(perp), max(0,-perp)];
            end
            Pssa_simple = [PTemp(6-a:4), PTemp(1:5-a)];
        end
        function Pssa_sia = getPssa(obj, si, a)
            ns = obj.n_states;
            if any(si == 2:11)
                Pssa_sia = (1:ns == 1); % Return to initial state from cliff
            elseif si == 12
                Pssa_sia = (1:ns == 12); % Stay in end state
            else
                Pssa_simple = getSimplePssa(obj,a);
                sn = si + 12;
                se = si + 1;
                ss = si - 12;
                sw = si - 1;
                P_stay = 0;
                Pssa_sia = zeros(ns, 1);
                if sn > ns % If there is no north state
                    P_stay = P_stay + Pssa_simple(1);
                else
                    Pssa_sia(sn) = Pssa_simple(1);
                end
                if se > 12*ceil(si/12) % If there is no east state
                    P_stay = P_stay + Pssa_simple(2);
                else
                    Pssa_sia(se) = Pssa_simple(2);
                end
                if ss < 1 % If there is no south state
                    P_stay = P_stay + Pssa_simple(3);
                else
                    Pssa_sia(ss) = Pssa_simple(3);
                end
                if sw <= 12*floor((si-1)/12) % If there is no west state
                    P_stay = P_stay + Pssa_simple(4);
                else
                    Pssa_sia(sw) = Pssa_simple(4);
                end
                Pssa_sia(si) = P_stay;
            end
        end
    end
end
