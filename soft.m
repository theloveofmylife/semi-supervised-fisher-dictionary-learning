function y = soft(x,tau)
%ÈíãÐÖµº¯Êý
y = sign(x).*max(abs(x)-tau/2,0);