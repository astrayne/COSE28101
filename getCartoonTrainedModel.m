function [net, cartoonData, cartoonLabel] = getCartoonTrainedModel(pixel)
% clear all
% close all
% clc
[cartoonData, cartoonLabel] = getCartoonData(64);
figure;
totalNum = size(cartoonData,1);

perm = randperm(totalNum,20);
for i = 1:20
    subplot(4,5,i);
    imshow(cartoonData{perm(i)}(:,:));
end

%%
%labelCount = size(categories(cartoonLabel), 1);

%% Specify Training and Validation Sets

%set number of dataset

%split train and valid dataset randomly
% randInd = randperm(totalNum);
% for i=1:trainNum
%     trainData(:,:,i) = cartoonData{randInd(1,i)};
%     %trainLabel(i,1) = cartoonLabel(randInd(1,i));
% end
% for i=1:valNum
%     valData(:,:,i) = cartoonData{randInd(1,i+trainNum)};
%     %valLabel(i,1) = cartoonLabel(randInd(1,i+trainNum));
% end

%For stratified random sampling
cv = cvpartition(cartoonLabel, 'HoldOut', 0.5, 'Stratify', true);
cv_val = cvpartition(cartoonLabel(cv.test), 'HoldOut', 0.6, 'Stratify', true);

trainNum = sum(cv.training==1);
valNum = sum(cv.test==1);

trainLabel = zeros(trainNum,1);
valLabel = zeros(valNum,1);

trainLabelIndex = cv.training;
valLabelIndex = cv.test;
trainData = zeros(pixel, pixel, trainNum);
valData = zeros(pixel, pixel, valNum);
tempTrainData = cartoonData(cv.training);
tempValData = cartoonData(cv.test);
j=1;
k=1;
for i=1:totalNum
    if trainLabelIndex(i,1) == 1
        trainLabel(j,1) = cartoonLabel(i,1);
        j=j+1;
    end
    if valLabelIndex(i,1) == 1
        valLabel(k,1) = cartoonLabel(i,1);
        k=k+1;
    end
end
for i=1:trainNum
    trainData(:,:,i) = tempTrainData{i};
end
for i=1:valNum
    valData(:,:,i) = tempValData{i};
end
%trainLabel = cartoonLabel(cv.training);
%valLabel = cartoonLabel(cv_val.training);

% trainData = cell2mat(trainData);
% valData = cell2mat(valData);
trainData = reshape(trainData, [pixel,pixel,1,trainNum]);
valData = reshape(valData, [pixel,pixel,1,valNum]);


%% Define Network Architecture
% Define the convolutional neural network architecture.
layers = [
    imageInputLayer([pixel pixel 1])
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    reluLayer
    
    fullyConnectedLayer(128)
    reluLayer();
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer
    
    ];


options = trainingOptions('sgdm',...
    'MaxEpochs',2, ...
    'ValidationData',{valData,categorical(valLabel)},...
    'ValidationFrequency',30,...
    'Verbose',true,...
   'L2Regularization',0.05,...
    'Plots','training-progress','ExecutionEnvironment','Parallel');

%% Train Network Using Training Data
net = trainNetwork(trainData, categorical(trainLabel),layers,options);

%% Classify Validation Images and Compute Accuracy
predictedLabels = classify(net,valData);

accuracy = sum(predictedLabels == categorical(valLabel))/numel(valLabel)