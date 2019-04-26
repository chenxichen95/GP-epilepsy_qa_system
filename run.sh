#! /bin/bash
rootPath=$(pwd)

# run demo_data
cd ${rootPath}/demo_data
python ./load_demo_data.py
if [ $? -ne 0 ];then
  echo "##############################################################"
  echo " can't run demo_data/load_demo_data.py" 
  echo "##############################################################"
  exit
fi

cd ${rootPath}/textPreprocessing
# run text preprocessing
python ./main_run.py
if [ $? -ne 0 ];then
  echo "##############################################################"
  echo " can't run textPreprocessing/main_run.py" 
  echo "##############################################################"
  exit
fi


cd ${rootPath}/trainWord2vec
# run text preprocessing
python ./main_run.py
if [ $? -ne 0 ];then
  echo "##############################################################"
  echo " can't run trainWord2vec/main_run.py" 
  echo "##############################################################"
  exit
fi

batch="20"
epoch="10"
hidden_units="80"
hops="3"
embeddingSize="150"
dropout="0.0"
lr="0.001"
display_steps="1"

cd ${rootPath}/model_build_train_dev_test
# build model, train and dev model, test model
python ./main_run.py --batch ${batch} --epoch ${epoch} --hidden_units ${hidden_units} --hops ${hops} --embeddingSize ${embeddingSize} --dropout ${dropout} --lr ${lr} --display_steps ${display_steps}
if [ $? -ne 0 ];then
  echo "##############################################################"
  echo " can't run model_build_train_dev_test/main_run.py" 
  echo "##############################################################"
  exit
fi

