function [data, labels] = getCartoonData(pixel)
    people = ["aia","bonnie","jules","malcom","mery","ray"];
    emotions = ["anger","disgust","fear","joy","neutral","sadness", "surprise"];
    totalNum = 0;
    for person = people
        for emotion = emotions
            fpath = strcat('./data/', person, '/', person, '_', emotion);
            files = dir([char(fpath), '/*.png']);
            filesNum = size(files,1);
            totalNum = totalNum + filesNum;
        end
    end
    data = cell(totalNum,1);
    labels = categorical(totalNum, 1);
    fileIdx = 1;
    for person = people
        emotionIdx = 0;
        for emotion = emotions
            fpath = strcat('./data/', person, '/', person, '_', emotion);
            files = dir([char(fpath), '/*.png']);
            filesNum = size(files,1);
            for i=1:filesNum
                img = imread(strcat(files(i).folder, '/', files(i).name));
                data{fileIdx} = imresize(rgb2gray(img), [pixel pixel]);
                labels(fileIdx, 1) = categorical(emotionIdx);
                % labels(fileIdx, 1) = emotion;
                fileIdx = fileIdx + 1;
            end
            emotionIdx = emotionIdx+1;
        end
    end
    labels = removecats(labels);
    
    
