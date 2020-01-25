classdef NeuralNetwork < handle
    properties
        T;
        B1;
        B2;
        W1;
        W2;
        input_count;
        hidden_count;
        output_count;
        training_size;
        
        epsilon = 0.001;
        alpha = 0.7;
        beta = 0.3;
        
    end
    methods
        function this = NeuralNetwork(w1, w2, b1, b2)
            this.W1(:,:,1) = w1;
            this.W2(:,:,1) = w2;
            this.B1 = b1;
            this.B2 = b2;
            
            this.input_count = size(w1(:,:,1),2) - 1;
            this.hidden_count = size(w1(:,:,1),1);
            this.output_count = size(w2(:,:,1),1);
            
            this.epsilon = 0.001;
            this.alpha = 0.7;
            this.beta = 0.3;
            
        end
        function y = feedForward(Y, x)
            X = x;
            % Hidden Layer
            V1 = Y.W1(:,:,end) * X';
            Y1 = [1 arrayfun(@sigmoid,V1)'];
            
            % Output layer
            V2 = Y.W2(:,:,end)* Y1';
            Y2 = arrayfun(@sigmoid,V2);
            
            y = {Y1, Y2};
        end
        function y = evaluate(this, x) 
            X = [1, x];
            % Hidden Layer
            V1 = this.W1(:,:,end) * X';
            Y1 = [1 arrayfun(@sigmoid,V1)'];
            
            % Output layer
            V2 = this.W2(:,:,end)* Y1';
            Y2 = arrayfun(@sigmoid,V2);
            
            y = Y2;
        end
        function EES = train(this, T)
            this.training_size = size(T(:,1), 1);
            this.training_size
            d = T(:,(this.input_count+1):end);
            epoch = 1;
            while(true)
                for i = 1:this.training_size
                    currentX = [1, T(i,1:this.input_count)];
                    Y = feedForward(this, currentX);
                    
                    Y1 = cell2mat(Y(1));
                    Y2 = cell2mat(Y(2));
                    
                    %getting the error
                    E(i,:) = d(i,:) - Y2';
                    e=E(i,:);
                    
                    %Backpropagation
                    delta_out = e.*Y2'.*(ones(1, this.output_count)-Y2');
                    
                    phi_prime = Y1.*(ones(1, this.hidden_count+1,1)-Y1);
                    delta_hidden = phi_prime.*(sum(this.W2(:,:,end).*delta_out'));
                    
                    %Updating Weights
                    change1 = this.alpha * delta_hidden' * currentX;
                    change2 = this.alpha * delta_out' * Y1;
                    
                    change1 = change1(2:end,:);
                    if(epoch > 1)
                        change1 = change1 + (this.beta .* (this.W1(:,:,end-1) - this.W1(:,:,end-2)));
                        change2 = change2 + (this.beta .* (this.W2(:,:,end-1) - this.W2(:,:,end-2)));
                    end
                    
                    this.W1(:,:,end+1) = this.W1(:,:,end)+change1;
                    this.W2(:,:,end+1) = this.W2(:,:,end)+change2;
                    
                    %Optional: updating B1 and B2
                    this.B1 = this.W1(:,1,end);
                    this.B2 = this.W2(:,1,end);
                    
                end
                
                ES = E .* E;
                
                if(epoch>1)
                    PREV_EES = EES;
                    EES = sum(ES(:))/2;
                    if(abs(PREV_EES - EES) < this.epsilon)
                        break;
                    end
                else
                    EES = sum(ES(:))/2;
                end
                %Part A ends in one epocj
%                 break;
                epoch= epoch+1;
            end
            this.W1 = this.W1(:,:,end);
            this.W2 = this.W2(:,:,end);
        end
    end
end