#!/usr/bin/env bash

NAME=uproot4env
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt

if [[ -f $NAME/bin/activate ]]; then
  echo "uproot4env already installed. Run \`source $NAME/bin/activate\` to activate"
  exit 1
fi

source $LCG/setup.sh
python -m venv --copies $NAME
source $NAME/bin/activate
LOCALPATH=$(python -c 'import sys; prefix=sys.prefix.split("/")[-1]; print(f"./{prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')
export PYTHONPATH=${LOCALPATH}:$PYTHONPATH
python -m pip install uproot4 scikit-learn scipy awkward1
sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/' $NAME/bin/*
sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
sed -i "2a source ${LCG}/setup.sh" $NAME/bin/activate
sed -i "3a export PYTHONPATH=${LOCALPATH}:\$PYTHONPATH" $NAME/bin/activate
FILE=${NAME}.tar.gz
if test -f "$FILE"; then
    rm ${NAME}.tar.gz
fi
tar -zcf ${NAME}.tar.gz ${NAME}
