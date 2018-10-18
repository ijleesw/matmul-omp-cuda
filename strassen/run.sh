#!/bin/bash
arr=(512 1024 2048)

for i in "${arr[@]}"
do
	echo "DIM=$i"
	echo ""

	echo "Single-Thread :"
	g++ -O2 -DDIM=$i -DTHRESHOLD=32 main.cpp -o test ; ./test
	echo ""
	
	echo "Multi-Thread :"
	g++ -O2 -DDIM=$i -DTHRESHOLD=128 main.cpp -o test -Xpreprocessor -fopenmp -lomp -DOMP ; ./test
	echo ""
done

echo "Test done."
rm ./test