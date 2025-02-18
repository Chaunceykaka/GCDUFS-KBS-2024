function [Hv,obj]= diversity(fea,alpha,beta,gamma,lammbda,n,v,c,SE)
[v1,v2]=size(fea);
seed = 109612;
randn('seed',seed), rand('seed',seed);
C=rand(n);
C(C < 0) = 0;
C = C - diag(diag(C));
V = rand(n, c);
Hv=cell(v1,v2);
Dv=cell(v1,v2);
Kv=cell(v1,v2);
Lv=cell(v1,v2);
Diag1=cell(v1,v2); %%for H
d=zeros(v2,1);
sumL = zeros(n);
sumD = zeros(n);
T = ones(n,n)*(1/n)*(-1) + eye(n);
K_complement = zeros(n,n);
MaxIter=20;
obj=zeros(1,MaxIter);

for num = 1:v
    fea{num}=fea{num}';
    d(num)=size(fea{num},1);
    Hv{num}=randn(d(num),SE);
    Dv{num}=rand(n);
    Dv{num}(Dv{num} < 0) = 0;
    Dv{num} = Dv{num} - diag(diag(Dv{num}));
    Diag1{num}=eye(d(num));
    sumD = sumD+Dv{num};

    Sv = constructW_PKN(fea{num});
    Lv{num} = diag(sum(Sv))-(Sv+Sv')/2;
    sumL = sumL + Lv{num};
end
%% Iter
for iter=1:MaxIter
    %%update H
        for num=1:v
            K=fea{num}-fea{num}*(C+Dv{num});
            eigenmatrix=K*K'+gamma*Diag1{num};
            [Hv{num}, ~, ~]=eig1(eigenmatrix, SE, 0);

            Hi=sqrt(sum(Hv{num}.*Hv{num},2)+eps);
            diagonal=0.5./Hi;
            Diag1{num}=diag(diagonal);
        end

    %%update C
        Kh=zeros(n);
        KhD=zeros(n);
        for num=1:v
            Kv{num}=fea{num}'*Hv{num}*Hv{num}'*fea{num};
            KhD=KhD+Kv{num}*Dv{num};
            Kh=Kh+Kv{num};
        end
        sylvester_CA = Kh + alpha*eye(n) + eye(n);
        sylvester_CB = lammbda*sumL;
        sylvester_CC = Kh - KhD + (V*V'-sumD/v);
        C = sylvester(sylvester_CA,sylvester_CB,sylvester_CC);

        C(C < 0) = 0;
        C = C - diag(diag(C));

    %%update D
    for num=1:v
        K_complement = K_complement*0;
        Kv{num}=fea{num}'*Hv{num}*Hv{num}'*fea{num};
        for k=1:v
            if (k==num)
                continue;
            end
            K_complement =  K_complement + T*Dv{k}'*Dv{k}*T;
        end
        sylvester_DA = Kv{num} + eye(n)/v;
        sylvester_DB = beta*K_complement;
        sylvester_DC = Kv{num} - Kv{num}*C + (V*V'-C)/v;
        Dv{num} = sylvester(sylvester_DA,sylvester_DB,sylvester_DC);

        Dv{num}(Dv{num} < 0) = 0;
        Dv{num} = Dv{num} - diag(diag(Dv{num}));
    end

        sumD=zeros(n);
        for num=1:v
            sumD = sumD + Dv{num};
        end

    %%updata V

        A = ((C+sumD/v)+(C+sumD/v)')/2;
        V = V .* (0.5 + (A*V) ./ (2*(V*V')*V));

        V(V < 0) = 0;

    %%obj
    sumobj=0;
    for num=1:v
        K_complement = K_complement*0;
        for k=1:v
            if (k==num)
                continue;
            end
            K_complement =  K_complement + T*Dv{k}'*Dv{k}*T;
        end
        sumobj=sumobj ...
              +norm(Hv{num}'*fea{num}-Hv{num}'*fea{num}*(C+Dv{num}),'fro')^2 ...
              +beta*trace(Dv{num}*K_complement*Dv{num}')...
              +gamma*trace(Hv{num}'*Diag1{num}*Hv{num}) ...
              +lammbda*trace(C*Lv{num}*C')...
              +1/v*norm(C+Dv{num}-V*V','fro')^2;
    end
    sumobj=sumobj + alpha*(norm(C,'fro'))^2;
    obj(iter)=real(sumobj);
end

end