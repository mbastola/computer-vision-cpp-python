#! /bin/bash

folder="/home/manil/cse559A/PackageProject_1/ImageSets/ROC/$1" #graf/yosemite
img1=""
img2=""
if [ "$1" == "graf" ];then
    img1="img1.ppm"
    img2="img2.ppm"
else
    img1="Yosemite1.jpg"
    img2="Yosemite2.jpg"
fi
featureFile1="${folder}/1.f"
featureFile2="${folder}/2.f"
featureFile_mb1="${folder}/1_mb.f"
featureFile_mb2="${folder}/2_mb.f"
homographyFile="${folder}/H1to2p"
resultFolder="${folder}/results/"
cmd="/home/manil/cse559A/PackageProject_1/FeaturesSkel/Features"
args="computeFeatures ${folder}/${img1} ${featureFile1} 2"
${cmd} ${args}
args="computeFeatures ${folder}/${img2} ${featureFile2} 2"
${cmd} ${args}
args="computeFeatures ${folder}/${img1} ${featureFile_mb1} 3"
${cmd} ${args}
args="computeFeatures ${folder}/${img2} ${featureFile_mb2} 3"
${cmd} ${args}
args="roc ${featureFile1} ${featureFile2} ${homographyFile} 1 roc1.txt
 auc1.txt"
${cmd} ${args}
args="roc ${featureFile1} ${featureFile2} ${homographyFile} 2 roc2.txt
 auc2.txt"
${cmd} ${args}
args="roc ${featureFile_mb1} ${featureFile_mb2} ${homographyFile} 1 roc_mb1.txt auc_mb1.txt"
${cmd} ${args}
args="roc ${featureFile_mb1} ${featureFile_mb2} ${homographyFile} 2 roc_mb2.txt auc_mb2.txt"
${cmd} ${args}
gnuplot plot.roc.txt
mv "roc1.txt ${resultFolder}"
mv "roc2.txt ${resultFolder}"
mv "roc_mb1.txt ${resultFolder}"
mv "roc_mb2.txt ${resultFolder}"
mv "auc1.txt ${resultFolder}"
mv "auc2.txt ${resultFolder}"
mv "auc_mb1.txt ${resultFolder}"
mv "auc_mb2.txt ${resultFolder}"
