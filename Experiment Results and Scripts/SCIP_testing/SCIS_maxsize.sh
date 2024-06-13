#!/bin/sh
#1. To get the results in Table 4
for VAR in 2 3 4 5
do
# java -jar -Xmx4G SCIS.jar 4 acq_253.txt,earn_253.txt,crude.txt,trade.txt Reuters3_$VAR 0.02,0.02,0.02,0.02 1 $VAR 0.05,0.05,0.05,0.05 T_Reuters3.txt 0.5 MC 11 >> Reuters3_SCIS.out
# java -jar -Xmx4G SCIS.jar 5 rec.sport.hockey.txt,rec.motorcycles.txt,soc.religion.christian.txt,rec.sport.baseball.txt,sci.crypt.txt News_$VAR 0.02,0.02,0.02,0.02,0.02 1 $VAR 0.05,0.05,0.05,0.05,0.05 T_News.txt 0.5 MC 1 >> News_SCIS.out
#java -jar -Xmx4G SCIS.jar 3 student.txt,faculty.txt,course.txt WebKB_$VAR 0.02,0.02,0.02 1 $VAR 0.05,0.05,0.05 T_WebKB.txt 0.5 MC 11 >> WebKB_SCIS.out
# java -jar -Xmx4G SCIS.jar 4 USER6_s.txt,USER8_s.txt,USER4_s.txt,USER5_s.txt Unix$VAR 0.02,0.02,0.02,0.02 1 $VAR 0.05,0.05,0.05,0.05 T_Unix.txt 0.5 MC 1 >> Unix_SCIS.out
#java -jar -Xmx4G SCIS.jar 2 class0_s.txt,class2_s.txt Robot$VAR 0.02,0.02 1 $VAR 0.05,0.05 T_Robot.txt 0.5 MC 11 >> Robot_SCIS.out
# auslan2
# java -jar -Xmx4G SCIS.jar 10 4.txt,9.txt,7.txt,6.txt,5.txt,2.txt,1.txt,8.txt,3.txt,10.txt  Homo$VAR 0.02,0.02 1 3 0.05,0.05 T_auslan2.txt 0.05 MC 11 >> auslan2_SCIS.out
# aslbu
java -jar -Xmx4G SCIS.jar 7 209.txt,191.txt,195.txt,199.txt,218.txt,210.txt,203.txt  Homo$VAR 0.02,0.02 1 3 0.05,0.05 T_aslbu.txt 0.05 MC 11 >> aslbu_SCIS.out

done
