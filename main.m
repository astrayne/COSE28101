pixel = 64;
[cartoonNet, cartoonData, cartoonLabel] = getCartoonTrainedModel(pixel);
[faceData, faceLabel] = getFaceData(pixel);
faceDataNum = size(faceData, 1);
faceTestData = zeros(pixel, pixel, faceDataNum);
faceTestDataLabel = zeros(faceDataNum,1);
for i=1:faceDataNum
    faceTestData(:,:,i) = faceData{i};
    faceTestDataLabel(i,1) = faceLabel(i);
end
faceTestData = reshape(faceTestData, [pixel,pixel,1,faceDataNum]);
predictedLabels = classify(cartoonNet,faceTestData);
accuracy = sum(predictedLabels == categorical(faceTestDataLabel))/numel(faceTestDataLabel)

[ckData, ckLabel] = newTest(pixel);
faceDataNum = size(ckData, 1);
faceTestData = zeros(pixel, pixel, faceDataNum);
faceTestDataLabel = zeros(faceDataNum,1);
for i=1:faceDataNum
    faceTestData(:,:,i) = ckData{i};
    faceTestDataLabel(i,1) = ckLabel(i);
end
faceTestData = reshape(faceTestData, [pixel,pixel,1,faceDataNum]);
predictedLabels = classify(cartoonNet,faceTestData);
accuracy = sum(predictedLabels == categorical(faceTestDataLabel))/numel(faceTestDataLabel)