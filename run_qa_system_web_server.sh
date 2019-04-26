#! /bin/bash

rootPath=$(pwd)

cd ${rootPath}/epilepsy_qa_system_web

while getopts "m:h" opt;do
  case $opt in
    m)
      model=${OPTARG}
      ;;
    h)
      echo "-m modelPath"
      exit
      ;;
    \?)
      echo "use -m"
      exit
      ;;
  esac
done
python ./myServer.py --model ${model}
if [ $? -ne 0 ]; then
  echo "can't run epilepsy_qa_system_web/myServer.py"
  exit
fi

