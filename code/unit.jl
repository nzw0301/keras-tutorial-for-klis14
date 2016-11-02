srand(1112)  # reproducibility

sigmoid(u) = 1/(1+exp(-u))

x = Array(-5:4)
w = rand(10)

println(sigmoid(dot(x, w)))
