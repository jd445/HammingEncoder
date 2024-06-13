import os
# subprocess is used to run the command line
import subprocess
# sys is used to get the command line arguments
import sys
# time is used to get the current time
import time
import timeit



# get the command line arguments
# java -jar MiSeRe.jar -input:aslbu.dat -output:res -run:5s -extract:1024
datasets = ['aslbu', 'context'	,'robot'	,'epitope'	,'skating'	,'question'	,'unix'	,'Gene'	,'reuters']
datasets = ["pioneer"]
for dataset in datasets:
    nowtime =timeit.default_timer()
    subprocess.run("java -jar MiSeRe.jar -input:datasets/training-test-data/MiSeRe_data/{}_training_fold_0.text -output:res -run:10s -extract:16".format(dataset), shell=True)
    endtime = timeit.default_timer()
    file = open('time.txt','a')
    file.write('dataset: {} time: {}s \n'.format(dataset,endtime-nowtime))
    file.close()



