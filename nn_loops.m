clear;
clc;

T = csvread('training.csv');
input_count = 3;
output_count = 2;
d = T(:,(input_count+1):end);
B1 = csvread('b1.csv');
W1 = [B1, csvread('w1.csv')];
B2 = csvread('b2.csv');
W2 = [B2, csvread('w2.csv')];

for i = 1:size(T(:,1), 1)
    currentX = [1, T(i,1:input_count)];
    % Hidden Layer
    V1 = zeros(size(W1(:,1), 1),1);
    Y1 = zeros(size(W1(:,1), 1),1);
    for j = 1:size(W1(:,1), 1)
        V1(j) = currentX * W1(j,:)';
        Y1(j) = sigmoid(V1(j));
    end;
    Y1 = [1;Y1];
    
    % Output layer
    V2 = zeros(size(W2(:,1), 1),1);
    Y2 = zeros(size(W2(:,1), 1),1);
    for j = 1:size(W2(:,1), 1)
        V2(j) = Y1' * W2(j,:)';
        Y2(j) = sigmoid(V2(j));
    end;
    
    %getting the error
    E(i,:) = d(i,:) - Y2';
    e=E(i,:)';
    
    %Backpropagation
    delta_out = e.*Y2.*(ones(2,1)-Y2);
    phi_prime = Y1.*(ones(12,1)-Y1);
    delta_hidden = phi_prime.*sum((delta_out.*W2))';
    
    change1 = delta_hidden.*Y1;
    W1 = W1+change1(2:end);
    
    change2 = delta_out.*Y2;
    W2 = W2+change2;
end;