#!/usr/bin/env bash
set -e
set -x

name=$1
data=$2
root=exp/nl2sql/$name

for split in dev
do
  fpred=$root/${split}_pred.txt
  feval=$root/${split}_eval.txt
  ftest=$root/${split}_scores.txt

  python predict.py $root/best.tar ${data}/${split}.json --tables ${data}/tables.json --db data/database --dcache cache/bert --output $fpred

  python eval_scripts/evaluation.py --gold ${data}/${split}_gold.sql --pred $fpred --etype match --table ${data}/tables.json --db data/database > $feval

  python predict.py $root/best.tar ${data}/${split}.json --tables ${data}/tables.json --db data/database --dcache cache/bert --output ${fpred}.avg --resumes $root/best.*.tar

  python eval_scripts/evaluation.py --gold ${data}/${split}_gold.sql --pred ${fpred}.avg --etype match --table ${data}/tables.json --db data/database > ${feval}.avg

  grep "exact match" $feval > $ftest
  grep "exact match" ${feval}.avg > ${ftest}.avg
  echo $split
  cat $ftest
  cat ${ftest}.avg
done
