function y = soft(x,tau)
%����ֵ����
y = sign(x).*max(abs(x)-tau/2,0);