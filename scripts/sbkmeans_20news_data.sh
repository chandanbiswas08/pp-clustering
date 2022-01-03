#!/bin/bash
export MAVEN_OPTS="-Xmx51200m -XX:MaxPermSize=12800m"
if [ $# -lt 10 ]
then
        echo "Usage: $0 <num vecs> <num dimensions> <num ref classes> <num predicted classes (K)> <estimate signatures of centroids (true/false)> <use projections (true/false)> <sbminhash(true/false)> <sbdim> <estimate centroid using sampling(true/false)> <estimate centroid in R_d(true/false)>  <estimate centroid assining random bit(true/false)>  <num bin(m)>  <save.sbvects(true/false)>  <loadSBVecs(true/false)><historyWeight(Real in [0 1])>"
        exit
fi
echo "num_vecs dim component K estimate_signature use projections use sbitminhash $1 $2 $3 $4 $5 $6 $7"

DATAFILE=$1
NUMVECS=$2
DIM=$3
NUMCLASSES=$4
K=$4
ITERS_KMEANS=10

PROPFILE=propfile/20news_18828_10.$4.$5.$6.$7.sbdim_$8.$9.$10.$11.properties
OUTFILE=kmeansout/sbkmeans.out.20news.$2.$3.$4.$5.$6.$7.sbdim_$8.$9.$10.$11



# create the data properties file

cat > $PROPFILE  << EOF1
datafile=$DATAFILE
syntheticdata.numsamples=$NUMVECS
syntheticdata.numgaussians=$NUMCLASSES
vec.numdimensions=$DIM
syntheticdata.outdir=./syntheticdata/
kmeans.numclusters=$K
kmeans.iterations=$ITERS_KMEANS
kmeans.outfile=$OUTFILE
estimate.sum.signatures=$5
average.centroid.estimation=$6
clustering.by.sbminhash=$7
sb.numdimensions=$8
estimate.New.Centroids.Usnig.Sampling=$9
estimate.New.Centroids.In.R_d=$10
estimate.New.Centroids.By.RandomBit=$11
SBVecClustering.In.Euclidean.Space=$12
num.bin=$13
save.sbvects=$14
loadSBVecs=$15
history.weight=$16
gt_output=20news2
EOF1


echo "K-means clustering..."
mvn exec:java -Dexec.mainClass="ibm.research.drl.lshkmeans.SBKMeansClusterer" -Dexec.args="$PROPFILE"

#exit

cat $DATAFILE | awk '{print $NF}' > 20news1
paste $OUTFILE 20news1 > 20news2

echo "Evaluating o/p..."
mvn exec:java -Dexec.mainClass="ibm.research.drl.lshkmeans.ClustEval" -Dexec.args="$PROPFILE"

rm 20news1
rm 20news2

