clc;
clear variables;
close all;

faceDetector=vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
resolutn = [25, 50, 100, 150];
kernals = ["linear", "rbf", "polynomial"];

% load('faces.mat')
% load('class.mat');
load('path_final');
load('gender_final');
total_male = length(gender_final(gender_final == 1));
total_female = length(gender_final(gender_final == 0));
% total_male = 3;
% total_female = 1;
results(1,:) = ["Resolution", "Kernal Function", "Male Class Accuracy", "Female Class Accuracy"];
%%
k = 2;
for s = 1:size(resolutn,2)
    res = strcat(num2str(resolutn(1,s)),'x',num2str(resolutn(1,s)));
    res_mat = strcat(res,'.mat');
    for m = 1:size(kernals,2)
        name = strcat('SVMModel_',kernals(1,m),res_mat);
        load(name);

        male = 0;
        female = 0;

        for i = 1:5
           image_path = path_final{1,i};
           I = imread(image_path);

           if (size(I,3)==3)
               I = rgb2gray(I);
           end

           BB = step(faceDetector,I);
           if(~isempty(BB))
               face = I(BB(1,2):BB(1,2)+BB(1,3),BB(1,1):BB(1,1)+BB(1,4));
               face = imresize(face,[resolutn(1,s),resolutn(1,s)]);
               temp = reshape(face,1,numel(face));

               [label,score] = predict(SVMModel,double(temp));

               if label == 1
                   if label == gender_final(1,i)
                       male = male + 1;
                   end
               else
                   if label == gender_final(1,i)
                       female = female + 1;
                   end
               end
           end
        end
        male_class_acc = (male/total_male)*100;
        female_class_acc = (female/total_female)*100;
        results(k,:) = [res, kernals(1,m), num2str(male_class_acc), num2str(female_class_acc)];
        k = k+1;
    end
end

% Accuracy = k/size(class,1)

%%

% I = imread('face.jpg');
% figure, imshow(I);
% if (size(I,3)==3)
%        I = rgb2gray(I);
% end
% 
% BB = step(faceDetector,I); % Detect faces
% 
% for i = 1:size(BB,1)
%     
%     face = I(BB(i,2):BB(i,2)+BB(i,3),BB(i,1):BB(i,1)+BB(i,4));
%     face = imresize(face,[50,50]);
%     temp = reshape(face,1,numel(face));
% 
%     [label,score] = predict(SVMModel,double(temp));
%     
%     if label == 1
%         gender = 'Male';
%     else
%         gender = 'Female';
%     end
%     Gender{i} = gender;
% end
% 
% I = imread('face.jpg');
% iimg = insertObjectAnnotation(I, 'rectangle', BB, Gender); %Annotate detected faces.
% figure;
% imshow(iimg); 
% title('Detected face');


