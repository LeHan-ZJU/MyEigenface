%function FaceRecognition(TestFace)
%基于Eigenface
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
%计算所有脸的均值
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
[V,~] = eig(C);%D为特征值构成的对角阵，每个特征值对应于V矩阵中列向量（也正是其特征向量）
U=zeros(46000,20);
figure;suptitle('特征脸');
%根据C的特征向量计算协方差矩阵的。再计算特征脸并显示出来
for p=1:20 
    U(:,p)=Differ*V(:,p);
    FeatureFace=reshape(U(:,p),230,200);
    FeatureFace=uint8(FeatureFace);
    subplot(4,5,p);
    imshow(FeatureFace);
    direct=[cd,'\TestSet\FeatureFaces\'];   %保存特征脸
    imwrite(FeatureFace,[direct, 'FeatureFace',sprintf('%d',p),'.gif']);
end
%计算样本集内每个脸通过特征脸的权重表示,OmegaAll每一列为每个脸对应的特征脸表示
OmegaAll=zeros(20,20);  %19行（每个人的权重值表9个特征值的权重，20列（20个人）
for s=1:20
    for r=1:20
        OmegaAll(r,s)=U(:,r)'*Differ(:,s);
    end
end
%根据上一个程序testEigenface的结果来识别训练集外的人脸
%TestFace=imread('TestFace4.jpg');  %TF:TestFace
VTestFace=double(TestFace(:));
DiFace=VTestFace-MeanAll(:,1);  %DiFace为待识别脸的列向量与平均脸之差
OmegaTF=zeros(20,1);%19行，1列
for t=1:20
    OmegaTF(t,1)=U(:,t)'*DiFace;
end
Distance=zeros(20,2);%distance第一列依次为输入图与20个样本的距离，第二列为其对应的序号
for u=1:20
    Distance(u,2)=u;
    Distance(u,1)=norm(OmegaTF-OmegaAll(:,u));
end
A1=Distance;
A1(:,1)=sort(Distance(:,1));     %A1第一列存放距离的升序排列
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
num=MatrixAll(1,4);     % [~,num]=min(Distance(:,1));
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