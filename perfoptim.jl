using BenchmarkTools
using ThreadsX
using LinearAlgebra

"""
unstable sum, can be optimized by:
- changing the result variable data type to float
- ThreadsX map after certain size of input array
"""
function unstable_sum(X)
    #= r = 0. 
    for x ∈ X
        r += x
    end
    return r =#
    # ThreadsX sum:
    return ThreadsX.sum(X) 
end

X = rand(100_000);
@btime unstable_sum($X)


"""
outside memory allocation example. Assume A, B square
"""
function matmul(A, B)
    return A*B
end

A = rand(10_000, 10_000); B = rand(10_000, 10_000);
R = zeros(size(A));
@btime matmul($A, $B)
@btime mul!($R, $A, $B) # no memory allocation, slightly faster

"""
using in place function example
"""
function normalize_allocate(v)
    return v/norm(v)
end

v = rand(10_000)*10;
@btime normalize_allocate(v)
@btime normalize!(v) # faster and no memory allocation but replaces v := v/norm(v)

"""
optimize complex function
"""
function calculate(v)
    return v .* sin.(v) .+ cos.(v)
end

function calculate!(r, v)
    for (i,el) ∈ enumerate(v)
        @inbounds r[i] = el * sin(el) + cos(el)
    end
end

v = rand(1_000_000)*10;
@btime calculate($v)
r = zeros(size(v));
@btime calculate!($r,$v) # this makes 0 allocation


"""
more complex functions
"""
function apply_filter(img::Matrix{Float64})
    m, n = size(img)
    result = zeros(m, n)
    for i in 2:m-1
        for j in 2:n-1
            result[i, j] = (img[i-1, j] + img[i+1, j] +
                            img[i, j-1] + img[i, j+1] -
                            4 * img[i, j]) / 4
        end
    end
    return result
end

function apply_filter!(result::Matrix{Float64}, img::Matrix{Float64})
    m, n = size(img)
    for j in 2:n-1
        for i in 2:m-1
            result[i, j] = (img[i-1, j] + img[i+1, j] +
                            img[i, j-1] + img[i, j+1] -
                            4 * img[i, j]) / 4
        end
    end
end

A = rand(1000, 1000)
F = zeros(size(A));
@btime apply_filter($A)
@btime apply_filter!($F, $A)  #no allocation, much faster