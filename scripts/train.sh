#!/bin/bash
# Tune the algorithm's parameters

#export LC_NUMERIC="en_US.UTF-8"

# noise levels
sigmas=(10 20 40 60)

# fixed parameters
pszs=(3 4 5 6 7 8 10 12)
wszs=(15 20 25 30 35 40)

# number of trials
ntrials=1

# test images
images=(\
gray_alley.png \
gray_book.png \
)
unused=(\
gray_building1.png \
gray_building2.png \
gray_computer.png \
gray_dice.png \
gray_flowers1.png \
gray_flowers2.png \
gray_gardens.png \
gray_girl.png \
gray_hallway.png \
gray_man1.png \
gray_man2.png \
gray_plaza.png \
gray_statue.png \
gray_street1.png \
gray_street2.png \
gray_traffic.png \
gray_trees.png \
gray_valldemossa.png \
gray_yard.png \
)

# seq folder
#sf='/mnt/nas-pf/'
sf='/home/pariasm/denoising/data/images/tune/'

output=${1:-"trials"}

export OMP_NUM_THREADS=1

# we assume that the binaries are in the same folder as the script
BIN=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
echo $BIN

for ((i=0; i < $ntrials; i++))
do
	# randomly draw noise level and parameters

	# noise level
	s=$(awk -v M=2 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}'); s=${sigmas[$s]}

	# patch size
	p=$(awk -v M=8 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}'); p1=${pszs[$p]}
	p=$(awk -v M=8 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}'); p2=${pszs[$p]}

	# search window
	w=$(awk -v M=6 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}'); w1=${wszs[$w]}
	w=$(awk -v M=6 -v s=$RANDOM 'BEGIN{srand(s); print int(rand()*(M+1))}'); w2=${wszs[$w]}

	# number of similar patches
	n1min=$((p1 * p1)); n1max=$((2 * n1min))
	n2min=$((p2 * p2)); n2max=$((2 * n2min))
	n1=$(awk -v m=$n1min -v M=$n1max -v s=$RANDOM 'BEGIN{srand(s); print int(m + rand()*(M-m+1))}')
	n2=$(awk -v m=$n2min -v M=$n2max -v s=$RANDOM 'BEGIN{srand(s); print int(m + rand()*(M-m+1))}')

	echo $n1 $n2

	# other parameters
	f1=$(awk -v m=0.8 -v M=1.2 -v s=$RANDOM 'BEGIN{srand(s); print m + rand()*(M-m)}')
	t2=$(awk -v m=4.0 -v M=32. -v s=$RANDOM 'BEGIN{srand(s); print m + rand()*(M-m)}')

	echo $s $n1min $n1max $n2min $n2max
	echo $s $p1 $w1 $n1 $f1 $p2 $w2 $n2 $t2

	ss=$(printf "%02d" $s)
	sp1=$(printf "%02d" $p1)
	sw1=$(printf "%02d" $w1)
	sn1=$(printf "%03d" $n1)
	sf1=$(printf "%04.2f" $f1)
	sp2=$(printf "%02d" $p2)
	sw2=$(printf "%02d" $w2)
	sn2=$(printf "%03d" $n2)
	st2=$(printf "%04.2f" $t2)

	trialfolder="$output/s${ss}-p${sp1}w${sw1}n${sn1}f${sf1}-p${sp2}w${sw2}n${sn2}f${st2}"

	echo $trialfolder

	params=" -p1 $sp1 -w1 $sw1 -n1 $sn1 -f1 $sf1 -p2 $sp2 -w2 $sw2 -n2 $sn2 -t2 $st2"

	echo $params

	mmse=0
	nimages=${#images[@]}
	if [ ! -d $trialfolder ]
	then
		for im in ${images[@]}
		do
			echo "$BIN/nlbayes-gt.sh ${sf}${im} $s $trialfolder \"$params\""
			mse=$($BIN/nlbayes-gt.sh ${sf}${im} $s $trialfolder  "$params")
			echo $mse $mmse
			mmse=$(plambda -c "$mmse $mse $nimages / +")
			echo $mse $mmse
		done
	fi
	
	printf "$ss $sp1 $sw1 $sn1 $sf1 $sp2 $sw2 $sn2 $st2  %7.4f\n" $mmse >> $output/table

	if [ -d $trialfolder/deno.tif ]
	then
		rm $trialfolder/*
	fi

done
