clear;
clc;

%Reading CSV data.
T = csvread('training.csv');
B1 = csvread('b1.csv');
W1(:,:,1) = [B1, csvread('w1.csv')];
B2 = csvread('b2.csv');
W2(:,:,1) = [B2, csvread('w2.csv')];

%Calculating helper variables
input_count = size(W1(:,:,1),2) - 1;
hidden_count = size(W1(:,:,1),1);
output_count = size(W2(:,:,1),1);
training_size = size(T(:,1), 1);

%constants
alpha = 0.7;
beta = 0.3;

d = T(:,(input_count+1):end);

for i = 1:training_size
    currentX = [1, T(i,1:input_count)];
    
    % Hidden Layer
    V1 = W1(:,:,end) * currentX';
    Y1 = [1 arrayfun(@sigmoid,V1)'];
    
    % Output layer
    V2 = W2(:,:,end)* Y1';
    Y2 = arrayfun(@sigmoid,V2);
    
    %getting the error
    E(i,:) = d(i,:) - Y2';
    e=E(i,:);
    
    %Backpropagation
    
    delta_out = e.*Y2'.*(1-Y2');
    delta_hidden = Y1.*(1-Y1) .* (sum(W2(:,:,end).*delta_out'));
    
    %Calculating the change
    change1 = alpha * delta_hidden' * currentX;
    change2 = alpha * delta_out' * Y1;
    change1 = change1(2:end,:);
    
    %Adding the Momentum to the change after the first sample
    if(i > 1)
        change1 = change1 + (beta .* (W1(:,:,end) - W1(:,:,end-1)));
        change2 = change2 + (beta .* (W2(:,:,end) - W2(:,:,end-1)));
    end
    
    %Updating the weights (including biases)
    W1(:,:,end+1) = W1(:,:,end)+change1;
    W2(:,:,end+1) = W2(:,:,end)+change2;
end;

%Output variables
B1 = W1(:,1,end);
B2 = W2(:,1,end);
W1 = W1(:,2:end,end);
W2 = W2(:,2:end,end);

SE = E .* E;
SSE = sum(SE(:));
