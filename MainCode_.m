function MainCode_()
global opt
close all
% Reading 2D input
Number_of_slice=randi([0 400]);
A=imread("originaldata.tif",Number_of_slice);
pixel_length=3;%nm
figure; imshow(A)

% Segmenting 2D input
A=imgaussfilt(A,1);
binary_value=0.65;
B=imbinarize(A,'adaptive','sensitivity',binary_value);
B=medfilt2(B,[3,3]);
opt.B0=B;
figure; imagesc(B)

% 2D to 3D Reconstruction
X=FIND(B);
S=[size(A,1),size(A,2),400];
C=gen(X,S);
volumeViewer(C);
end

function AT=gen(X,S)
% global opt
A=rand(S); A=imgaussfilt3(A,X(1).*4+.01,'padding','circular');   A=normal(A);
A1=A;
A=rand(S); A=imgaussfilt3(A,X(2).*10+.01,'padding','circular');  A=normal(A);
A2=A;
W=zeros(S); W(end,:,:)=1; W=bwdist(W);
W=normal(W.^(10));
AT=A1.*W+A2.*(1-W);
AT=flatout(AT,.015);
AT=AT>quantile(AT,X(3)*.46+.06);
AT(S(1)/4:end,:,:)=imclose(AT(S(1)/4:end,:,:),sph(1+X(4)*3,3));
end

function F=feature(A)
if numel(size(A))==3; A=A(:,:,ceil(size(A,3)/2)); end
S=[size(A,1),size(A,2),8];
F1=feature0(A(1:round(size(A,1)/5),:));
F2=feature0(A(end-round(size(A,1)/2):end,:));
F=[F1 F2]+1;
end

function X=FIND(A)
global opt
S=[size(A,1),size(A,2),5];
F0=feature(A);
opt.iter=0;
opt.F0=F0;
opt.S=S;
opt.MinE=1e6;
x1 = optimizableVariable('x1',[0,1],'Type','real');
x2 = optimizableVariable('x2',[0,1],'Type','real');
x3 = optimizableVariable('x3',[0,1],'Type','real');
x4 = optimizableVariable('x4',[0,1],'Type','real');
MAXEVAL=150;
results = bayesopt(@Err,[x1,x2,x3,x4],...
    'ExplorationRatio',.7,...
    'AcquisitionFunctionName','expected-improvement',...
    'MaxObjectiveEvaluations',MAXEVAL,...
    'MaxTime',300,...
    'GPActiveSetSize',300,'PlotFcn',...
    {@plotMinObjective,@plotAcquisitionFunction});
X=table2array(results.XAtMinObjective);
end


function E=Err(X)
global opt
opt.iter=opt.iter+1;
if istable(X); X=table2array(X); end
AT=gen(X,opt.S);
F=feature(AT);
E=mean(abs(opt.F0-F)./opt.F0);
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
subplot(2,3,1); imagesc(opt.B0);  axis equal square tight; title('Original');
subplot(2,3,2); imagesc(AT(:,:,ceil(size(AT,3)/2)));  axis equal square tight; title('Current reconstruction');
if opt.MinE>E
    opt.MinE=E;
    opt.BestF=F;
    opt.BestX=X;
    subplot(2,3,3);
    imagesc(AT(:,:,ceil(size(AT,3)/2)));  axis equal square tight; title('Best reconstruction');
end
subplot(2,3,4); cla; plot(opt.F0); hold on;  plot(opt.BestF,'color','g'); legend({'Original','Best reconstruction'})
xlabel('Feature ID'); ylabel('Feature value');
subplot(2,3,5); hold on; scatter(opt.iter,opt.MinE,'k'); title(['Current error= ' num2str(E)]);
xlabel('Iterations'); ylabel('Average relative error'); box on;
subplot(2,3,6); cla; bar([opt.BestX ; X]'); ylim([0,1]); legend({'Best reconstruction','Current reconstruction'})
xlabel('Latent space variables'); ylabel('Latent space values');
drawnow;
end

function [A]=normal(A)
A=double(A);
M1=min(A(:)); M2=max(A(:));
if M1==M2; return; end
A=(A-M1)./(M2-M1);
end

function [A]=flatout(A,prc)
low=quantile(A,prc);
high=quantile(A,1-prc);
A(A<low)=low;
A(A>high)=high;
end

function [SE]=sph(r,dim)
if nargin==1; dim=3; end
if dim==3
    s=ceil(2*r+1); SE=zeros(s,s,s);
    for I=1:s
        for J=1:s
            for K=1:s
                D=sqrt((I-(r+1))^2+(J-(r+1))^2+(K-(r+1))^2);
                if D<=r; SE(I,J,K)=1; end;
            end
        end
    end
end
if dim==2
    s=ceil(2*r+1); SE=zeros(s,s);
    for I=1:s
        for J=1:s
            D=sqrt((I-(r+1))^2+(J-(r+1))^2);
            if D<=r; SE(I,J)=1; end;
        end
    end
end
end

function [F]=feature0(A)
B1=bwdist(A); Q1=B1(:); Q1(Q1==0)=[]; B2=bwdist(~A); Q2=B2(:); Q2(Q2==0)=[];
F=[fliplr(histcounts(Q1,1:29/50:30)), histcounts(Q2,1:29/50:30)]./prod(size(A));
end

function [q]=quantile(x,p)
x=sort(x(:)); id=ceil(p.*max(size(x)));
id(id==0)=1;
q=x(id);
end