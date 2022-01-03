unzip RealData/20_news_group/20_news_group.vec.zip -d RealData/20_news_group/


# save 𝜙-transformed vectors and global statistics
# sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 true true false 32 false false false false 10 true false 0
# sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 true true false 64 false false false false 10 true false 0
# sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 true true false 128 false false false false 10 true false 0
# sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 true true false 256 false false false false 10 true false 0
# sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 true true false 512 false false false false 10 true false 0


# K-means
echo "\n\n---------------------K-means---------------------\n\n"
python3 main.py real 0 0 kmeans 0.1 0 0 20news 20 0
# F-score	ARI	NMI	0.3240	0.2840	0.5014
echo "\n\n\n\n\n"


# LSH-partition
echo "\n\n---------------------LSH-partition--------------------\n\n"
sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 false false true 256 false false false false 10 false true 0
# F-score	ARI	NMI	0.0968	0.0019	0.0679
echo "\n\n\n\n\n"


# 𝜙-Kmeans
echo "\n\n---------------------𝜙-Kmeans--------------------\n\n"
sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 false false false 256 false false false true 10 false false 0
# F-score	ARI	NMI	0.3599	0.3232	0.5149
echo "\n\n\n\n\n"


# E2HK-means
echo "\n\n---------------------E2HK-means--------------------\n\n"
sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 false false false 256 false true false false 10 false false 0
# F-score	ARI	NMI	0.3411	0.3048	0.4559
echo "\n\n\n\n\n"


# PPK-means
echo "\n\n---------------------PPK-means--------------------\n\n"
python3 main.py h2e 256 512 kmeans 0.1 0 0 20news 20 0
# F-score	ARI	NMI	0.3810	0.3458	0.5065
echo "\n\n\n\n\n"


#H-𝜓-Kmeans
echo "\n\n---------------------H-𝜓-Kmeans--------------------\n\n"
python3 main.py h-psi 256 512 kmeans 0.1 10 0.000001 20news 20 0
# F-score	ARI	NMI	0.3364	0.2999	0.4702
echo "\n\n\n\n\n"


# 𝜓-Kmeans-NP
echo "\n\n---------------------𝜓-Kmeans-NP--------------------\n\n"
python3 main.py h2e-psi-noprior 256 512 kmeans 0.1 20 0.000001 20news 20 0
# F-score	ARI	NMI	0.3700	0.3352	0.5120
echo "\n\n\n\n\n"


# 𝜓-Kmeans
echo "\n\n---------------------𝜓-Kmeans--------------------\n\n"
python3 main.py h2e-psi 256 512 kmeans 0.1 20 0.000001 20news 20 0
# F-score	ARI	NMI	0.3958	0.3630	0.5194
echo "\n\n\n\n\n"


# Sensitivity of encoding dimension m
# 𝜙-Kmeans
for hdim in 32 64 128 256
do
	sh scripts/sbkmeans_20news_data.sh RealData/20_news_group/20_news_group.vec 18828 200 20 false false false $hdim false false false true 10 false false 0
done
# F-score	ARI	NMI	0.1700	0.1252	0.2618
# F-score	ARI	NMI	0.2443	0.2026	0.3749
# F-score	ARI	NMI	0.3242	0.2857	0.4699
# F-score	ARI	NMI	0.3599	0.3232	0.5149



# PPK-means
for hdim in 32 64 128 256
do
	python3 main.py h2e $hdim 512 kmeans 0.1 0 0 20news 20 0
done
# F-score	ARI	NMI	0.1578	0.1039	0.2397
# F-score	ARI	NMI	0.2828	0.2404	0.3848
# F-score	ARI	NMI	0.3341	0.2961	0.4688
# F-score	ARI	NMI	0.3810	0.3458	0.5065


# H-𝜓-Kmeans
for hdim in 32 64 128 256
do
	python3 main.py h-psi $hdim 512 kmeans 0.1 10 0.000001 20news 20 0
done
# F-score	ARI	NMI	0.1138	0.0653	0.1648
# F-score	ARI	NMI	0.2059	0.1630	0.3005
# F-score	ARI	NMI	0.2899	0.2511	0.4053
# F-score	ARI	NMI	0.3364	0.2999	0.4702

# 𝜓-Kmeans-NP
python3 main.py h2e-psi-noprior 32 512 kmeans 0.1 10 0.000001 20news 20 0
python3 main.py h2e-psi-noprior 64 512 kmeans 0.1 60 0.00001 20news 20 0
python3 main.py h2e-psi-noprior 128 512 kmeans 0.1 20 0.00001 20news 20 0
python3 main.py h2e-psi-noprior 256 512 kmeans 0.1 20 0.000001 20news 20 0
# F-score	ARI	NMI	0.1266	0.0748	0.1802
# F-score	ARI	NMI	0.2469	0.2036	0.3495
# F-score	ARI	NMI	0.3071	0.2668	0.4425
# F-score	ARI	NMI	0.3700	0.3352	0.5120

# 𝜓-Kmeans
python3 main.py h2e-psi 32 512 kmeans 0.1 10 0.000001 20news 20 0
python3 main.py h2e-psi 64 512 kmeans 0.1 60 0.00001 20news 20 0
python3 main.py h2e-psi 128 512 kmeans 0.1 20 0.00001 20news 20 0
python3 main.py h2e-psi 256 512 kmeans 0.1 20 0.000001 20news 20 0
# F-score	ARI	NMI	0.1381	0.0861	0.2003
# F-score	ARI	NMI	0.2813	0.2403	0.3886
# F-score	ARI	NMI	0.3385	0.3005	0.4665
# F-score	ARI	NMI	0.3958	0.3630	0.5194


# Sensitivity of embedding dimension p
# H-𝜓-Kmeans
for EMB_DIM in 64 128 256 512
do
	python3 main.py h-psi 256 $EMB_DIM kmeans 0.1 10 0.000001 20news 20 0
done
# F-score	ARI	NMI	0.2644	0.2246	0.3640
# F-score	ARI	NMI	0.2883	0.2499	0.4027
# F-score	ARI	NMI	0.3459	0.3100	0.4597
# F-score	ARI	NMI	0.3371	0.3007	0.4707



# 𝜓-Kmeans-NP
for EMB_DIM in 64 128 256 512
do
	python3 main.py h2e-psi-noprior 256 $EMB_DIM kmeans 0.1 10 0.000001 20news 20 0
done
# F-score	ARI	NMI	0.3439	0.3070	0.4536
# F-score	ARI	NMI	0.3496	0.3138	0.4723
# F-score	ARI	NMI	0.3168	0.2726	0.4712
# F-score	ARI	NMI	0.3747	0.3402	0.5114


# 𝜓-Kmeans
for EMB_DIM in 64 128 256 512
do
	python3 main.py h2e-psi 256 $EMB_DIM kmeans 0.1 10 0.000001 20news 20 0
done
# F-score	ARI	NMI	0.3579	0.3190	0.4830
# F-score	ARI	NMI	0.3152	0.2700	0.4797
# F-score	ARI	NMI	0.3805	0.3465	0.5187
# F-score	ARI	NMI	0.3958	0.3630	0.5193


# Sensitivity of tau
# 𝜓-Kmeans
for tau in 0.01 0.02 0.05 0.1 0.15
do
	python3 main.py h2e-psi 256 512 kmeans $tau 10 0.000001 20news 20 0
done

# F-score	ARI	NMI	0.3957	0.3629	0.5193
# F-score	ARI	NMI	0.3860	0.3523	0.5046
# F-score	ARI	NMI	0.3957	0.3629	0.5193
# F-score	ARI	NMI	0.3958	0.3630	0.5194



python3 main.py real 0 0 kmeans 0.1 0 0 20news 10 0
python3 main.py real 0 0 kmeans 0.1 0 0 20news 20 0
python3 main.py real 0 0 kmeans 0.1 0 0 20news 30 0
python3 main.py real 0 0 kmeans 0.1 0 0 20news 40 0
# F-score	ARI	NMI	0.3085	0.2550	0.5088
# F-score	ARI	NMI	0.3240	0.2840	0.5014
# F-score	ARI	NMI	0.2884	0.2540	0.4981
# F-score	ARI	NMI	0.2894	0.2611	0.4998


python3 main.py h2e-psi 256 512 kmeans 0.1 20 0.000001 20news 10 0
python3 main.py h2e-psi 256 512 kmeans 0.1 20 0.000001 20news 20 0
python3 main.py h2e-psi 256 512 kmeans 0.1 20 0.000001 20news 30 0
python3 main.py h2e-psi 256 512 kmeans 0.1 20 0.000001 20news 40 0


# F-score	ARI	NMI	0.3269	0.2759	0.4803
# F-score	ARI	NMI	0.3958	0.3630	0.5194
# F-score	ARI	NMI	0.3277	0.2947	0.4969
# F-score	ARI	NMI	0.3154	0.2889	0.4882


