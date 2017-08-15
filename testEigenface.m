%function[MeanAll,U,OmegaAll]= testEigenface
%参照博客http://blog.csdn.net/smartempire/article/details/21406005
clear all
close all
exm=imread('face1.jpg');
[m,n]=size(exm);
l=m*n;
VectorAll=zeros(l,20);  %初始化大矩阵(长为图片拉成列向量的长度,宽为图片数)
figure;suptitle('样本集');
for i=1:20
    name1=sprintf('face%d.jpg',i);
    A= imread(name1);
    subplot(4,5,i);
    imshow(A);
    VectorAll(:,i)=A(:); %图片转化为列向量
end

%矩阵Differ对应原博客中的矩阵A，C为原博客中的矩阵L
MeanAll=zeros(l,20);
for k=1:l
    MeanAll(k,:)=mean(VectorAll(k,:));
end
Face=reshape(MeanAll(:,1),m,n);
Face=uint8(Face);
direct=[cd,'\TestSet\'];   %保存平均脸
imwrite(Face,[direct, 'MeanFace.gif']);
figure;
imshow(Face);title('所有图片的平均脸')
Differ=VectorAll-MeanAll; %每个样本脸与平均脸做差
%Differ=bsxfun(@minus, VectorAll, mean(VectorAll,2));
% Differ0=VectorAll-MeanAll;
% Differ=Differ0+ones(l,20)*127.5;
figure;suptitle('原图与平均脸的差值');
for j=1:20
   B0=reshape(Differ(:,j),230,200); 
   B=uint8(B0);
   subplot(4,5,j);
   imshow(B);
   direct=[cd,'\TestSet\DifferFace\'];   %保存平均脸
   imwrite(B,[direct, 'DifferFace',sprintf('%d',j),'.gif']);
end
%计算协方差矩阵Differ*Differ'的其特征值与特征向量
C=cov(Differ);%Differ'*Differ;%先计算C的特征向量。
[V,D] = eig(C);  %D为特征值构成的对角阵，每个特征值对应于V矩阵中列向量（也正是其特征向量）
%U=zeros(46000,20);%U用来存储所有的特征脸列向量
U=zeros(46000,20);
figure;suptitle('特征脸');
%根据C的特征向量计算协方差矩阵的。再计算特征脸并显示出来
for p=1:20 
%     for q=1:20
%         U(:,p)=V(q,p)*Differ(:,q)+U(:,p);
%     end
    U(:,p)=Differ*V(:,p);
    FeatureFace=reshape(U(:,p),230,200);
    FeatureFace=uint8(FeatureFace);
    subplot(4,5,p);
    imshow(FeatureFace);
    direct=[cd,'\TestSet\FeatureFaces\'];   %保存平均脸
   imwrite(FeatureFace,[direct, 'FeatureFace',sprintf('%d',p),'.gif']);
end
%计算样本集内每个脸通过特征脸的权重表示,OmegaAll每一列为每个脸对应的特征脸表示
% OmegaAll=zeros(20,20);
% for r=1:20
%     OmegaAll(:,r)=U(:,r)'*Differ(:,r);
% end
OmegaAll=zeros(20,20);  %19行（每个人的权重值表9个特征值的权重，20列（20个人）
for s=1:20
    for r=1:20
        OmegaAll(r,s)=U(:,r)'*Differ(:,s);
    end
end

