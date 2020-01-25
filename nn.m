clear;
clc;

T = csvread('training.csv');
B1 = csvread('b1.csv');
W1(:,:,1) = [B1, csvread('w1.csv')];
B2 = csvread('b2.csv');
W2(:,:,1) = [B2, csvread('w2.csv')];

input_count = size(W1(:,:,1),2) - 1;
hidden_count = size(W1(:,:,1),1);
output_count = size(W2(:,:,1),1);
training_size = size(T(:,1), 1);

epsilon = 0.001;
alpha = 0.7;
beta = 0.3;

d = T(:,(input_count+1):end);

epoch = 1;
while(true)
    for i = 1:training_size
        currentX = [1, T(i,1:input_count)];
        
        % Hidden Layer
        V1 = W1(:,:,i+(epoch-1)*training_size) * currentX';
        Y1 = [1 arrayfun(@sigmoid,V1)'];
        
        % Output layer
        V2 = W2(:,:,i+(epoch-1)*training_size)* Y1';
        Y2 = arrayfun(@sigmoid,V2);
        
        %getting the error
        E(i,:) = d(i,:) - Y2';
        e=E(i,:);
        
        %Backpropagation
        delta_out = e.*Y2'.*(ones(1, output_count)-Y2');
        
        phi_prime = Y1.*(ones(1, hidden_count+1,1)-Y1);
        delta_hidden = phi_prime.*(sum(W2(:,:,end).*delta_out'));
        
        %Updating Weights
        change1 = alpha * delta_hidden' * currentX;
        change2 = alpha * delta_out' * Y1;
        
        change1 = change1(2:end,:);
        if(i+(epoch-1)*training_size > 1) 
            change1 = change1 + (beta .* (W1(:,:,end) - W1(:,:,end-1)));
            change2 = change2 + (beta .* (W2(:,:,end) - W2(:,:,end-1)));
        end
        
        W1(:,:,end+1) = W1(:,:,end)+change1;
        W2(:,:,end+1) = W2(:,:,end)+change2;
        
        %Optional: updating B1 and B2
        B1 = W1(:,1,end);
        B2 = W2(:,1,end);
        
    end;
    ES = E .* E;

    if(epoch>1)
        PREV_EES = EES;
        EES = sum(ES(:))/2;
        if(abs(PREV_EES - EES) < epsilon)
            break;
        end
    else
        EES = sum(ES(:))/2
    end
    %Part A ends in one epocj
    break;
    epoch= epoch+1;
end
