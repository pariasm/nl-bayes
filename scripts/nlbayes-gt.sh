#!/bin/bash
# Evals vnlm using ground truth

IN=$1 # input image
SIG=$2 # noise standard dev.
OUT=$3 # output folder
PRM=$4 # denoiser parameters

# we assume that the binaries are in the same folder as the script
DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

mkdir -p $OUT

# error checking {{{1
if [ ! -f $IN ]
then
	echo ERROR: $IN not found
	exit 1
fi
 
# add noise {{{1
file=$(printf $OUT/"noisy.tif" $i)
if [ ! -f $file ]
then
	export SRAND=$RANDOM;
	awgn $SIG $IN $file
fi

# run denoising {{{1
$DIR/nlbayes $SIG $OUT"/noisy.tif" $OUT"/deno.tif" $PRM
 
# compute psnr {{{1
MSE=$(psnr.sh $IN $OUT/"deno.tif" m 0)
RMSE=$(plambda -c "$MSE sqrt")
PSNR=$(plambda -c "255 $RMSE / log10 20 *")

echo "RMSE $RMSE" >> $OUT/measures
echo "PSNR $PSNR" >> $OUT/measures
echo $MSE


# vim:set foldmethod=marker:
