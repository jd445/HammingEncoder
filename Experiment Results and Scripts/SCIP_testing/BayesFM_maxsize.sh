#!/bin/sh
#1. To get the results in Table 4
for VAR in 2 3 4 5
do
#java -jar -Xmx4G BayesFM.jar 4 acq_253.txt,earn_253.txt,crude.txt,trade.txt Reuters3_$VAR 0.02,0.02,0.02,0.02 1 $VAR 0.05,0.05,0.05,0.05 T_Reuters3.txt 0.5 MC >> Reuters3_BayesFM.out
#java -jar -Xmx4G BayesFM.jar 3 student.txt,faculty.txt,course.txt WebKB_$VAR 0.02,0.02,0.02 1 $VAR 0.05,0.05,0.05 T_WebKB.txt 0.5 MC >> WebKB_BayesFM.out
java -jar -Xmx4G BayesFM.jar 5 rec.sport.hockey.txt,rec.motorcycles.txt,soc.religion.christian.txt,rec.sport.baseball.txt,sci.crypt.txt News_$VAR 0.02,0.02,0.02,0.02,0.02 1 $VAR 0.05,0.05,0.05,0.05,0.05 T_News.txt 0.5 MC >> News_BayesFM.out
java -jar -Xmx4G BayesFM.jar 4 USER6_s.txt,USER8_s.txt,USER4_s.txt,USER5_s.txt Unix$VAR 0.02,0.02,0.02,0.02 1 $VAR 0.05,0.05,0.05,0.05 T_Unix.txt 0.5 MC >> Unix_BayesFM.out
#java -jar -Xmx4G BayesFM.jar 2 class0_s.txt,class2_s.txt Robot$VAR 0.02,0.02 1 $VAR 0.05,0.05 T_Robot.txt 0.5 MC >> Robot_BayesFM.out
done
