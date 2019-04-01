function sim = gaussianKernel(x1, x2, sigma)

x1 = x1(:); x2 = x2(:);
diff = sum((x1-x2).^2,1);
sim = exp(-0.5*sigma^(-2)*diff);
    
end
