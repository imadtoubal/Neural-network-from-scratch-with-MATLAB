clear;
T = csvread('training.csv');
B1 = csvread('b1.csv');
W1(:,:,1) = [B1, csvread('w1.csv')];
B2 = csvread('b2.csv');
W2(:,:,1) = [B2, csvread('w2.csv')];

NN = NeuralNetwork(W1, W2, B1, B2);
% feedForward(T(1:3), NN);
train(NN, T);
a = evaluate(NN, [-1.99200000000000,-1.67900000000000,-0.0680000000000000]);