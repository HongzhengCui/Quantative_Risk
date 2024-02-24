using LinearAlgebra
using Distributions
using Random
using BenchmarkTools
using Plots
using DataFrames
using CSV 


#Problem 1
#Read data and drop the column with dates
rets = CSV.read("Project/DailyReturn.csv",DataFrame)
filter!(r->!ismissing(r.SPY), rets)
nm = names(rets)
nm = nm[nm.!="Column1"]

for n in nm
    # println(n)
    if typeof(rets[1,n]) <:  InlineString
        println("Running $n")
        rets[!,n] = parse.(Float64,rets[:,n])
    end
end

#Function to calculate expoentially weighted covariance.  
function ewCovar(x,λ)
    m,n = size(x)
    w = Vector{Float64}(undef,m)

    #Remove the mean from the series
    xm = mean.(eachcol(x))
    for j in 1:n
        x[:,j] = x[:,j] .- xm[j]
    end

    #Calculate weight.  Realize we are going from oldest to newest
    for i in 1:m
        w[i] = (1-λ)*λ^(m-i)
    end
    #normalize weights to 1
    w = w ./ sum(w)

    #covariance[i,j] = (w # x)' * x  where # is elementwise multiplication.
    return (w .* x)' * x
end

function expW(m,λ)
    w = Vector{Float64}(undef,m)
    for i in 1:m
        w[i] = (1-λ)*λ^(m-i)
    end
    #normalize weights to 1
    w = w ./ sum(w)
    return w
end


function PCA_pctExplained(a)
    n = size(a,1)
    #Get Eigenvalues
    vals = reverse!(real.(eigvals(a)))

    #total Eigenvalues
    total = sum(vals)

    out = Vector{Float64}(undef,n)
    s = 0.0
    for i in 1:n
        s += vals[i]
        out[i] = s/total  #cumulative % of the total
    end
    return out
end

#Function tests
covar = ewCovar(Matrix(rets[!,nm]),0.97)
expl = PCA_pctExplained(covar)

pctExplained = DataFrame(:x=>[i for i in 1:(size(rets,2)-1)])
pctExplained[!,Symbol("λ=0.75")] = PCA_pctExplained(ewCovar(Matrix(rets[!,nm]),0.75))
pctExplained[!,Symbol("λ=0.85")] = PCA_pctExplained(ewCovar(Matrix(rets[!,nm]),0.85))
pctExplained[!,Symbol("λ=0.90")] = PCA_pctExplained(ewCovar(Matrix(rets[!,nm]),0.90))
pctExplained[!,Symbol("λ=0.95")] = PCA_pctExplained(ewCovar(Matrix(rets[!,nm]),0.95))
pctExplained[!,Symbol("λ=0.99")] = PCA_pctExplained(ewCovar(Matrix(rets[!,nm]),0.99))

cnames = names(pctExplained)
cnames = cnames[findall(x->x!="x",cnames)]

plot(pctExplained.x,Array(pctExplained[:,cnames]), label=hcat(cnames...), legend=:bottomright, title="% Explained by EigenValue")
#As lambda descreases, the percent explained by the first eigenvalues increases.  This is because more weight
#is added to the more recent observations.  The lower the lambda, the lower the rank of the matrix

#problem #2

function near_psd(a; epsilon=0.0)
    n = size(a,1)

    invSD = nothing
    out = copy(a)

    #calculate the correlation matrix if we got a covariance
    if count(x->x ≈ 1.0,diag(out)) != n
        invSD = diagm(1 ./ sqrt.(diag(out)))
        out = invSD * out * invSD
    end

    #SVD, update the eigen value and scale
    vals, vecs = eigen(out)
    vals = max.(vals,epsilon)
    T = 1 ./ (vecs .* vecs * vals)
    T = diagm(sqrt.(T))
    l = diagm(sqrt.(vals))
    B = T*vecs*l
    out = B*B'

    #Add back the variance
    if invSD !== nothing 
        invSD = diagm(1 ./ diag(invSD))
        out = invSD * out * invSD
    end
    return out
end
#Cholesky that assumes PSD
function chol_psd!(root,a)
    n = size(a,1)
    #Initialize the root matrix with 0 values
    root .= 0.0

    #loop over columns
    for j in 1:n
        s = 0.0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>1
            s =  root[j,1:(j-1)]'* root[j,1:(j-1)]
        end
  
        #Diagonal Element
        temp = a[j,j] .- s
        if 0 >= temp >= -1e-8
            temp = 0.0
        end
        root[j,j] =  sqrt(temp);

        #Check for the 0 eigan value.  Just set the column to 0 if we have one
        if 0.0 == root[j,j]
            root[j,(j+1):n] .= 0.0
        else
            #update off diagonal rows of the column
            ir = 1.0/root[j,j]
            for i in (j+1):n
                s = root[i,1:(j-1)]' * root[j,1:(j-1)]
                root[i,j] = (a[i,j] - s) * ir 
            end
        end
    end
end

#Helper functions from Notes
function _getAplus(A)
    vals, vecs =eigen(A)
    vals = diagm(max.(vals,0))
    return vecs * vals*vecs'
end

function _getPS(A,W)
    W05 = sqrt.(W)
    iW = inv(W05)
    return (iW * _getAplus(W05*A*W05) * iW)
end

function _getPu(A,W)
    Aret = copy(A)
    for i in 1:size(Aret,1)
        Aret[i,i] = 1.0
    end
    return Aret
end

function wgtNorm(A,W)
    W05 = sqrt.(W)
    W05 = W05 * A * W05
    return sum(W05 .* W05)
end

function higham_nearestPSD(pc,W=nothing, epsilon=1e-9,maxIter=100,tol=1e-9)

    n = size(pc,1)
    if W === nothing
        W = diagm(fill(1.0,n))
    end

    deltaS = 0

    Yk = copy(pc)
    norml = typemax(Float64)
    i=1

    while i <= maxIter
        # println("$i - $norml")
        Rk = Yk .- deltaS
        #Ps Update
        Xk = _getPS(Rk,W)
        deltaS = Xk - Rk
        #Pu Update
        Yk = _getPu(Xk,W)
        #Get Norm
        norm = wgtNorm(Yk-pc,W)
        #Smallest Eigenvalue
        minEigVal = min(real.(eigvals(Yk))...)

        # print("Yk: "); display(Yk)
        # print("Xk: "); display(Xk)
        # print("deltaS: "); display(deltaS)

        if abs(norm - norml) < tol && minEigVal > -epsilon
            # Norm converged and matrix is at least PSD
            break
        end
        # println("$norml -> $norm")
        norml = norm
        i += 1
    end
    if i < maxIter 
        println("Converged in $i iterations.")
    else
        println("Convergence failed after $(i-1) iterations")
    end
    return Yk
end

n=500
sigma=fill(0.9,(n,n))
for i in 1:n   
    sigma[i,i] = 1.0
end
sigma[1,2] = 0.7357
sigma[2,1] = 0.7357

W = diagm(fill(1.0,n)) #Identity matrix as weight

hpsd = higham_nearestPSD(sigma)
npsd = near_psd(sigma)
norm_hpsd = wgtNorm(hpsd-sigma,W)
norm_npsd = wgtNorm(npsd-sigma,W)
println("Distance near_psd()=$norm_npsd")
println("Distand higham_nearestPSD()=$norm_hpsd")

higam_times = @benchmark higham_nearestPSD(sigma)
near_times = @benchmark near_psd(sigma)

println("n=500")
println("Higam Took: $(mean(higam_times.times/1e9)) seconds")
println("Near_PSD Took: $(mean(near_times.times/1e9)) seconds")

#Do it again but with a bigger matrix
println("n=1000")

n=1000
sigma=fill(0.9,(n,n))
for i in 1:n   
    sigma[i,i] = 1.0
end
sigma[1,2] = 0.7357
sigma[2,1] = 0.7357

higam_times = @benchmark higham_nearestPSD(sigma)
near_times = @benchmark near_psd(sigma)

println("Higam Took: $(mean(higam_times.times/1e9)) seconds")
println("Near_PSD Took: $(mean(near_times.times/1e9)) seconds")

#Higham is much slower than Near_PSD but gets you to a matrix that is closer to the original
#You have to decide on the tradeoff.  For fast calculations where close is "good enough" then
#use near_psd.  If you need more precision and can wait, use Higham

#problem 3

#Normal Simulation Function:
function simulateNormal(N::Int64, cov::Array{Float64,2}; mean=[],seed=1234)

    #Error Checking
    n, m = size(cov)
    if n != m
        throw(error("Covariance Matrix is not square ($n,$m)"))
    end


    out = Array{Float64,2}(undef,(n,N))

    #If the mean is missing then set to 0, otherwise use provided mean
    _mean = fill(0.0,n)
    m = size(mean,1)
    if !isempty(mean)
        if n!=m
            throw(error("Mean ($m) is not the size of cov ($n,$n"))
        end
        copy!(_mean,mean)
    end


    # Take the root
    l = Array{Float64,2}(undef,n,n)
    chol_psd!(l,cov)

    # try
    #     l = Matrix(cholesky(cov).L)
    # catch e
    #     if isa(e, LinearAlgebra.PosDefException)
    #         # println("Matrix is not PD, assuming PSD and continuing.")
    #         l = copy(cov)
    #         chol_psd!(l,cov)
    #     else
    #         throw(e)
    #     end
    # end
    

    #Generate needed random standard normals
    Random.seed!(seed)
    d = Normal(0.0,1.0)

    rand!(d,out)

    #apply the standard normals to the cholesky root
    out = (l*out)'

    #Loop over itereations and add the mean
    for i in 1:n
        out[:,i] = out[:,i] .+ _mean[i]
    end
    out
end

#PCA
function simulate_pca(a, nsim; pctExp=1, mean=[],seed=1234)
    n = size(a,1)

    #If the mean is missing then set to 0, otherwise use provided mean
    _mean = fill(0.0,n)
    m = size(mean,1)
    if !isempty(mean)
        copy!(_mean,mean)
    end

    #Eigenvalue decomposition
    vals, vecs = eigen(a)
    vals = real.(vals)
    vecs = real.(vecs)
    #julia returns values lowest to highest, flip them and the vectors
    flip = [i for i in size(vals,1):-1:1]
    vals = vals[flip]
    vecs = vecs[:,flip]
    
    tv = sum(vals)

    posv = findall(x->x>=1e-8,vals)
    if pctExp < 1
        nval = 0
        pct = 0.0
        #figure out how many factors we need for the requested percent explained
        for i in 1:size(posv,1)
            pct += vals[i]/tv
            nval += 1
            if pct >= pctExp 
                break
            end
        end
        if nval < size(posv,1)
            posv = posv[1:nval]
        end
    end
    vals = vals[posv]

    vecs = vecs[:,posv]

    # println("Simulating with $(size(posv,1)) PC Factors: $(sum(vals)/tv*100)% total variance explained")
    B = vecs*diagm(sqrt.(vals))

    Random.seed!(seed)
    m = size(vals,1)
    r = randn(m,nsim)

    out = (B*r)'
    #Loop over itereations and add the mean
    for i in 1:n
        out[:,i] = out[:,i] .+ _mean[i]
    end
    return out
end

#warmup each function
sim = simulateNormal(100,covar)
sim = simulate_pca(covar,100,pctExp=.5)

pearson_cov = cov(Matrix(rets[!,nm]))
pearson_std = sqrt.(diag(pearson_cov))
pearson_cor = cor(Matrix(rets[!,nm]))

ewma_cov = ewCovar(Matrix(rets[!,nm]),0.97)
ewma_std = sqrt.(diag(ewma_cov))
ewma_cor = diagm(1 ./ ewma_std) * ewma_cov * diagm(1 ./ewma_std)


matrixType = ["EWMA", "EWMA_COR_PEARSON_STD", "PEARSON", "PEARSON_COR_EWMA_STD"]
simType = ["Full", "PCA=1", "PCA=0.75", "PCA=0.5"]

matrixLookup = Dict{String, Array{Float64,2}}()
matrixLookup["EWMA"] = ewma_cov
matrixLookup["EWMA_COR_PEARSON_STD"] = diagm(pearson_std) * ewma_cor * diagm(pearson_std)
matrixLookup["PEARSON"] = pearson_cov
matrixLookup["PEARSON_COR_EWMA_STD"] = diagm(ewma_std) * pearson_cor * diagm(ewma_std)

matrix = Vector{String}(undef,16)
simulation = Vector{String}(undef,16)
runtimes = Vector{Float64}(undef,16)
norms = Vector{Float64}(undef,16)

i=1
for sim in simType
    for mat in matrixType
        global i
        matrix[i] = mat
        simulation[i] = sim
        c = matrixLookup[mat]
        elapse = 0.0
        s = []
        if sim == "Full"
            st = time()
            for loops in 1:20
                s = simulateNormal(25000,c)
            end
            elapse = (time() - st)/20
        elseif sim =="PCA=1"
            st = time()
            for loops in 1:20
                s = simulate_pca(c,25000,pctExp=1)
            end
            elapse = (time() - st)/20
        elseif sim=="PCA=0.75"
            st = time()
            for loops in 1:20
                s = simulate_pca(c,25000,pctExp=.75)
            end
            elapse = (time() - st)/20
        else
            st = time()
            for loops in 1:20
                s = simulate_pca(c,25000,pctExp=.5)
            end
            elapse = (time() - st)/20
        end

        covar = cov(s)
        runtimes[i] = elapse
        norms[i] = sum( (covar-c).^2 )
        i = i+1
    end
end

outTable = DataFrame(:Matrix=>matrix,
                     :Simulation=>simulation,
                     :Runtime=>runtimes,
                     :Norm=>norms
                )
