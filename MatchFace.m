%function MatchFace(MeanAll,U,OmegaAll,TestFace)
%根据上一个程序testEigenface的结果来识别训练集外的人脸
TestFace=imread('TestFace7.jpg');  %TF:TestFace
VTestFace=double(TestFace(:));
DiFace=VTestFace-MeanAll(:,1);  %DiFace为待识别脸的列向量与平均脸之差
OmegaTF=zeros(20,1);
for t=1:20
    OmegaTF(t,1)=U(:,t)'*DiFace;
end
Distance=zeros(20,2);%distance第一列依次为输入图与20个样本的距离，第二列为其对应的序号
for u=1:20
    Distance(u,2)=u;
    Distance(u,1)=norm(OmegaTF-OmegaAll(:,u));
end
%A1第一列存放距离的升序排列
A1=Distance;
A1(:,1)=sort(Distance(:,1));
ser=zeros(20,1);
for w=1:20
    [ser(w,1),~]=find(Distance(:,1)==A1(w,1));
end
%MatrixAll前两列为Distance，第三列为距离升序排列，第四列为第三列各值原来的序号
MatrixAll=zeros(20,4);
MatrixAll(:,1)=Distance(:,1);
MatrixAll(:,2)=Distance(:,2);
MatrixAll(:,3)=A1(:,1);
MatrixAll(:,4)=ser;
%寻找距离最短的图像为识别结果，并输出
num=MatrixAll(1,4);
% [~,num]=min(Distance(:,1));
name2=sprintf('face%d.jpg',num);  
figure;
subplot(1,2,1),imshow(TestFace);
title('输入的人脸');
subplot(1,2,2),imshow(name2);
title('识别出最接近的人脸')
%将各图像依照距离从小到大的顺序输出
figure;suptitle('识别结果（从左到右，上到下相似度依次降低）');
for x=1:20
    name3=sprintf('face%d.jpg',MatrixAll(x,4)); 
    subplot(4,5,x),imshow(name3);
end    