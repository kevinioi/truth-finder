from liblinearutil import *
# Read data in LIBSVM format
y, x = svm_read_problem('../heart_scale')
m = train(y[:200], x[:200], '-c 4')
p_label, p_acc, p_val = predict(y[200:], x[200:], m)

# Construct problem in python format
# Dense data
y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# Sparse data
y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
prob  = problem(y, x)
param = parameter('-s 0 -c 4 -B 1')
m = train(prob, param)

# Other utility functions
save_model('heart_scale.model', m)
m = load_model('heart_scale.model')
p_label, p_acc, p_val = predict(y, x, m, '-b 1')
ACC, MSE, SCC = evaluations(y, p_label)

# Getting online help
help(train)