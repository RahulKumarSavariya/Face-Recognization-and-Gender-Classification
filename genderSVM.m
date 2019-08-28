clc;
clear variables;
close all;

faceDetector=vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
faces = [];

% [class, image_paths] = xlsread('D:\SEAS - AU\Course Work\Machine Learning\Assignments\04 Soft Biometrics\COSFIRE_Gender_recognition_1_0\Application\GENDER-FERET\TrainingSet.xlsx');
% class = image_paths(:,1);

load('path_final');
load('gender_final');

resolutn = [25, 50, 100, 150];
kernals = ["linear", "rbf", "polynomial"];

for s = 1:size(resolutn,2)
    k = 1;
    faces = [];
    for i = 1:5
       image_path = path_final{1,i};
       I = imread(image_path);

       if (size(I,3)==3)
           I = rgb2gray(I);
       end

       BB = step(faceDetector,I); % Detect faces
       if(~isempty(BB))
           face = I(BB(1,2):BB(1,2)+BB(1,3),BB(1,1):BB(1,1)+BB(1,4));
           face = imresize(face,[resolutn(1,s),resolutn(1,s)]);
           temp = reshape(face,1,numel(face));

           faces(k,:) = temp;
           class(k,1) = gender_final(1,i);
    %        trained_path{k,1} = image_path;
           k = k+1;
       end

    end
    
    res = strcat(num2str(resolutn(1,s)),'x',num2str(resolutn(1,s)),'.mat');
    for m = 1:size(kernals,2)
        SVMModel = fitcsvm(faces,class,'KernelFunction',char(kernals(1,m)));
        name = strcat('SVMModel_',kernals(1,m),res);
        save(char(name),'SVMModel')
    end
end

% SVMModel = fitcsvm(faces(1:4000,:),class(1:4000,:),'KernelFunction','linear');
% SVMModel = fitcsvm(faces,class,'KernelFunction','rbf');
% SVMModel = fitcsvm(faces,class,'KernelFunction','polynomial');

% CVSVMModel = crossval(SVMModel, 'kFold',10);
% classLoss = kfoldLoss(CVSVMModel)

% svmStruct = svmtrain(faces,class,'kernel_function','linear');

% I = imread('D:\SEAS - AU\Course Work\Machine Learning\Assignments\04 Soft Biometrics\COSFIRE_Gender_recognition_1_0\Application\GENDER-FERET\female\test_set\0.jpg');
% I = rgb2gray(I);
% BB = step(faceDetector,I); % Detect faces
% face = I(BB(1,2):BB(1,2)+BB(1,3),BB(1,1):BB(1,1)+BB(1,4));
% face = imresize(face,[50,50]);
% temp = reshape(face,1,numel(face));
   
% species = svmclassify(svmStruct,double(temp))