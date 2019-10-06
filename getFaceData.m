function [data, labels] = getFaceData(pixel)
    emotions = ["ANGRY","DISGUST","FEAR","HAPPY","NEUTRAL","SAD", "SURPRISE"];
    totalNum = 0;
    for emotion = emotions
        fpath = strcat('./jaffe/', emotion);
        files = dir([char(fpath), '/*.jpg']);
        filesNum = size(files,1);
        totalNum = totalNum + filesNum;
    end    
    data = cell(totalNum,1);
    labels = categorical(totalNum, 1);
    fileIdx = 1;
    emotionIdx = 0;
    for emotion = emotions
        fpath = strcat('./jaffe/', emotion);
        files = dir([char(fpath), '/*.jpg']);
        filesNum = size(files,1);
        for i=1:filesNum
            img = imread(strcat(files(i).folder, '/', files(i).name));
            data{fileIdx} = imresize(img, [pixel pixel]);
            labels(fileIdx, 1) = categorical(emotionIdx);
            fileIdx = fileIdx + 1;
        end
        emotionIdx = emotionIdx+1;
    end
    labels = removecats(labels);
    
    
