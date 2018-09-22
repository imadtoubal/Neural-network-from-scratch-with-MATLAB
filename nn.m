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

d = T(:,(input_count+1):end);

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
    change1 = delta_hidden' * currentX;
    W1 = W1+change1(2:end,:);
    
    change2 = delta_out'*Y1;
    W2 = W2+change2;
end;