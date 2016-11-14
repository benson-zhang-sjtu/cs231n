function grad_check_sparse(f, x, analytic_grad)
    num_checks = 10;
    h = 1e-5;
    
    for i=1:num_checks
        id = randi(numel(x));
        oldval = x(id);
        x(id) = oldval + h;
        [fxph,~] = f(x);
        x(id) = oldval - h;
        [fxmh,~] = f(x);
        x(id) = oldval;
        
        grad_numerical = (fxph - fxmh) / (2 * h);
        grad_analytic = analytic_grad(id);
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic));
        disp(sprintf('numerical: %f analytic: %f, relative error£º %e', grad_numerical, grad_analytic, rel_error));
    end
end