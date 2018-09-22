clear;
clc;

T = csvread('training.csv');
B1 = csvread('b1.csv');
W1 = [B1, csvread('w1.csv')];
B2 = csvread('b2.csv');
W2 = [B2, csvread('w2.csv')];

input_count = size(W1,2) - 1;
hidden_count = size(W1,1);
output_count = size(W2, 1);

epsilon = 0.001;
alpha = 0.7;
beta = 0.3;

d = T(:,(input_count+1):end);

epoch = 1;
while(true)
%         PREV_W1 = W1;
%         PREV_W2 = W2;
    
    for i = 1:size(T(:,1), 1)
        currentX = [1, T(i,1:input_count)];
        
        % Hidden Layer
        V1 = currentX * W1';
        Y1 = [1 arrayfun(@sigmoid,V1)];
        
        % Output layer
        V2 = Y1 * W2';
        Y2 = arrayfun(@sigmoid,V2);
        
        %getting the error
        E(i,:) = d(i,:) - Y2;
        e=E(i,:);
        
        %Backpropagation
        delta_out = e.*Y2.*(ones(1, output_count)-Y2);
        
        phi_prime = Y1.*(ones(1, hidden_count+1,1)-Y1);
        delta_hidden = phi_prime.*(sum(W2.*delta_out'));
        
        %Updating Weights
        change1 = alpha * delta_hidden' * currentX;
        change2 = alpha * delta_out' * Y1;
        
        change1 = change1(2:end,:);
%         if(epoch > 1) 
%             change1 = change1 + (beta .* (PREV_W1 - W1));
%             change2 = change2 + (beta .* (PREV_W2 - W2));
%         end
        
        W1 = W1+change1;
        W2 = W2+change2;
    end;
    ES = E .* E;
    if(epoch>1)
        PREV_EES = EES;
        EES = sum(ES(:))/2;
        if(abs(PREV_EES - EES) < epsilon)
            break;
        end
    else
        EES = sum(ES(:))/2;
    end
    EES
    epoch= epoch+1;
end