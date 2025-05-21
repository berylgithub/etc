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

"""
good ol' Gaussian Kernel. Maps 2 vectors → scalar
"""
function gaussian_kernel(u, v, σ)
    return exp(-norm(u-v)^2/2*σ^2)
end

# no vector allocation inside, since it's a reduce operation, just allocate a scalar
function gaussian_kernel_noalloc(u, v, σ)
    r = 0.
    for i ∈ eachindex(u)
        r += (u[i] - v[i])^2
    end
    return exp(-r/2*σ^2)
end

u = ones(100_000_000); v = zeros(100_000_000); σ = 1.;
@btime gaussian_kernel($u, $v, $σ)
@btime gaussian_kernel_noalloc($u, $v, $σ) # no allocation, much faster


"""
P.s.d matrix from Gaussian Kernel calls
"""
function generate_psdmatrix(A, B, σ)
    m = size(A,1); n = size(B,1);
    R = zeros(m,n)
    for i ∈ axes(A,1)
        for j ∈ axes(B,1)
            R[i,j] = gaussian_kernel_noalloc(A[i,:], B[j,:], σ)
        end
    end
    return R
end

function generate_psdmatrix!(R, A, B, σ)
    for i ∈ axes(A,1)
        for j ∈ axes(B,1)
            R[i,j] = gaussian_kernel_noalloc(@view(A[i,:]), @view(B[j,:]), σ) # @view is the key, it refers to the memory location rather than reallocating vectors
        end
    end
end

function generate_psdmatrix_threadsx(A, B, σ) # try ThreadsX map
    I = axes(A,1); J = axes(B, 1);
    IJ = Iterators.product(I, J) # product of two vectors of indices
    return ThreadsX.map(ij -> gaussian_kernel_noalloc(@view(A[ij[1],:]), @view(B[ij[2],:]), σ), IJ) # this will allocate, trade off between speed and memory
end


A = rand(10_000, 100); σ = 1.; 
R = zeros(size(A,1), size(A,1));
@btime generate_psdmatrix($A, $A, $σ)
@btime generate_psdmatrix!($R, $A, $A, $σ) # no allocation, faster
@btime generate_psdmatrix_threadsx($A, $A, $σ) # with ThreadsX, tradeoff between speed and memory, should be faster with higher number of threads (tested with 64 vs 1 threadnums)