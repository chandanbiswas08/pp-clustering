unzip RealData/mnist/mnist_vec_128.zip -d RealData/mnist/

# save 𝜙-transformed vectors and global statistics
# sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 true true false 32 false false false false 10 true false 0
# sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 true true false 64 false false false false 10 true false 0
# sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 true true false 128 false false false false 10 true false 0
# sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 true true false 256 false false false false 10 true false 0
# sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 true true false 512 false false false false 10 true false 0


# K-means
echo "\n\n---------------------K-means---------------------\n\n"
python3 main.py real 0 0 kmeans 0.1 0 0 mnist 10 0
# F-score	ARI	NMI	0.4295	0.3608	0.5181
echo "\n\n\n\n\n"

# LSH-partition
echo "\n\n---------------------LSH-partition--------------------\n\n"
sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 false false true 256 false false false false 10 false true 0
# F-score	ARI	NMI	0.2373	0.0961	0.2354
echo "\n\n\n\n\n"


# 𝜙-Kmeans
echo "\n\n---------------------𝜙-Kmeans--------------------\n\n"
sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 false false false 128 false false false true 10 false false 0
# F-score	ARI	NMI	0.3901	0.3209	0.4473
echo "\n\n\n\n\n"


# E2HK-means
echo "\n\n---------------------E2HK-means--------------------\n\n"
sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 false false false 256 false true false false 10 false false 0
# F-score	ARI	NMI	0.3646	0.2912	0.4336
echo "\n\n\n\n\n"


# PPK-means
echo "\n\n---------------------PPK-means--------------------\n\n"
python3 main.py h2e 256 0 kmeans 0.1 0 0 mnist 10 0
# F-score	ARI	NMI	0.4040	0.3361	0.4567
echo "\n\n\n\n\n"


# H-𝜓-Kmeans
echo "\n\n---------------------H-𝜓-Kmeans--------------------\n\n"
python3 main.py h-psi 256 512 kmeans 0.1 10 0.0001 mnist 10 0
# F-score	ARI	NMI	0.4174	0.3521	0.4533
echo "\n\n\n\n\n"


# 𝜓-Kmeans-NP
echo "\n\n---------------------𝜓-Kmeans-NP--------------------\n\n"
python3 main.py h2e-psi-noprior 256 512 kmeans 0.1 40 0.0007 mnist 10 0
# F-score	ARI	NMI	0.3987	0.3309	0.4529
echo "\n\n\n\n\n"


# 𝜓-Kmeans
echo "\n\n---------------------𝜓-Kmeans--------------------\n\n"
python3 main.py h2e-psi 256 512 kmeans 0.1 10 0.0001 mnist 10 0
# F-score	ARI	NMI	0.4370	0.3711	0.4607
echo "\n\n\n\n\n"


# Sensitivity of encoding dimension m
# 𝜙-Kmeans
for hdim in 32 64 128 256
do
	sh scripts/sbkmeans_mnist70k_data.sh RealData/mnist/mnist_vec_128 70000 128 10 false false false $hdim false false false true 10 false false 0
done
# F-score	ARI	NMI	0.3259	0.2437	0.3737
# F-score	ARI	NMI	0.3887	0.3188	0.4430
# F-score	ARI	NMI	0.3901	0.3209	0.4473
# F-score	ARI	NMI	0.3849	0.3152	0.4387


# PPK-means
for hdim in 32 64 128 256
do
	python3 main.py h2e $hdim 512 kmeans 0 100 0.0001 mnist 10 0
done
# F-score	ARI	NMI	0.3620	0.2867	0.4152
# F-score	ARI	NMI	0.3840	0.3145	0.4406
# F-score	ARI	NMI	0.3892	0.3207	0.4449
# F-score	ARI	NMI	0.4040	0.3361	0.4567


# H-𝜓-Kmeans
for hdim in 32 64 128 256
do
	python3 main.py h-psi $hdim 512 kmeans 0.1 10 0.0001 mnist 10 0
done
# F-score	ARI	NMI	0.3101	0.2287	0.3717
# F-score	ARI	NMI	0.3755	0.3043	0.4399
# F-score	ARI	NMI	0.3791	0.3070	0.4383
# F-score	ARI	NMI	0.4174	0.3521	0.4533




# 𝜓-Kmeans-NP
python3 main.py h2e-psi-noprior 32 512 kmeans 0.1 10 0.0004 mnist 10 0
python3 main.py h2e-psi-noprior 64 512 kmeans 0.1 10 0.0004 mnist 10 0
python3 main.py h2e-psi-noprior 128 512 kmeans 0.1 40 0.0005 mnist 10 0
python3 main.py h2e-psi-noprior 256 512 kmeans 0.1 40 0.0007 mnist 10 0
# F-score	ARI	NMI	0.2954	0.2130	0.3578
# F-score	ARI	NMI	0.3764	0.3061	0.4372
# F-score	ARI	NMI	0.3744	0.3046	0.4344
# F-score	ARI	NMI	0.3987	0.3309	0.4529



# 𝜓-Kmeans
python3 main.py h2e-psi 32 512 kmeans 0.1 200 0.0003 mnist 10 0
python3 main.py h2e-psi 64 512 kmeans 0.1 100 0.0005 mnist 10 0
python3 main.py h2e-psi 128 512 kmeans 0.1 40 0.0005 mnist 10 0
python3 main.py h2e-psi 256 512 kmeans 0.1 10 0.0001 mnist 10 0
# F-score	ARI	NMI	0.3705	0.2971	0.4254
# F-score	ARI	NMI	0.3853	0.3160	0.4429
# F-score	ARI	NMI	0.3892	0.3208	0.4449
# F-score	ARI	NMI	0.4370	0.3711	0.4607





# Sensitivity of embedding dimension p
# H-𝜓-Kmeans
for EMB_DIM in 64 128 256 512
do
        python3 main.py h-psi 256 $EMB_DIM kmeans 0.1 10 0.0001 mnist 10 0
done
# F-score	ARI	NMI	0.3485	0.2736	0.3980
# F-score	ARI	NMI	0.3667	0.2934	0.4203
# F-score	ARI	NMI	0.3960	0.3277	0.4426
# F-score	ARI	NMI	0.3855	0.3160	0.4377



# 𝜓-Kmeans-NP
python3 main.py h2e-psi-noprior 256 64 kmeans 0.1 10 0.0001 mnist 10 0
python3 main.py h2e-psi-noprior 256 128 kmeans 0.1 10 0.0001 mnist 10 0
python3 main.py h2e-psi-noprior 256 256 kmeans 0.1 10 0.0001 mnist 10 0
python3 main.py h2e-psi-noprior 256 512 kmeans 0.1 40 0.0007 mnist 10 0
# F-score	ARI	NMI	0.3812	0.3104	0.4304
# F-score	ARI	NMI	0.3486	0.2757	0.4005
# F-score	ARI	NMI	0.3878	0.3189	0.4323
# F-score	ARI	NMI	0.3961	0.3281	0.4508



# 𝜓-Kmeans
for EMB_DIM in 64 128 256 512
do
        python3 main.py h2e-psi 256 $EMB_DIM kmeans 0.1 10 0.0001 mnist 10 0
done
# F-score	ARI	NMI	0.3914	0.3222	0.4494
# F-score	ARI	NMI	0.4039	0.3361	0.4567
# F-score	ARI	NMI	0.4065	0.3386	0.4484
# F-score	ARI	NMI	0.4370	0.3711	0.4607

# Sensitivity of tau
# 𝜓-Kmeans
for tau in 0.01 0.02 0.05 0.1 0.15
do
	python3 main.py h2e-psi 256 512 kmeans $tau 10 0.0001 mnist 10 0
done
# F-score	ARI	NMI	0.4316	0.3687	0.4552
# F-score	ARI	NMI	0.4326	0.3665	0.4537
# F-score	ARI	NMI	0.4340	0.3669	0.4539
# F-score	ARI	NMI	0.4370	0.3711	0.4607
# F-score	ARI	NMI	0.4368	0.3709	0.4597


python3 main.py real 0 0 kmeans 0.1 0 0 mnist 5 0
python3 main.py real 0 0 kmeans 0.1 0 0 mnist 10 0
python3 main.py real 0 0 kmeans 0.1 0 0 mnist 15 0
python3 main.py real 0 0 kmeans 0.1 0 0 mnist 20 0
# F-score	ARI	NMI	0.3801	0.2797	0.4551
# F-score	ARI	NMI	0.4295	0.3608	0.5181
# F-score	ARI	NMI	0.3991	0.3401	0.5345
# F-score	ARI	NMI	0.3458	0.2933	0.5113



python3 main.py h2e-psi 256 512 kmeans 0.1 10 0.0001 mnist 5 0
python3 main.py h2e-psi 256 512 kmeans 0.1 10 0.0001 mnist 10 0
python3 main.py h2e-psi 256 512 kmeans 0.1 10 0.0001 mnist 15 0
python3 main.py h2e-psi 256 512 kmeans 0.1 10 0.0001 mnist 20 0
# F-score	ARI	NMI	0.3673	0.2688	0.3925
# F-score	ARI	NMI	0.4370	0.3711	0.4607
# F-score	ARI	NMI	0.3358	0.2741	0.4379
# F-score	ARI	NMI	0.2808	0.2250	0.4141

