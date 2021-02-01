#!/usr/bin/env bash
set -x
set -e
for s in 0 100 1000 10000
do
  fout=adapt/${s}.out.txt
  feval=adapt/${s}.eval.txt
  fres=adapt/${s}.res.txt
  python adapt.py best/spider_nl2sql.tar data/spider/dev.json --aug best/gen/spider_db/gen.pt best/gen/spider_db/gen2.pt --db data/database --tables data/tables.json --steps $s --output $fout
  python eval_scripts/evaluation.py --gold data/spider/dev_gold.sql --pred $fout --db data/database --table data/spider/tables.json --etype all > $feval
  grep -i "exact match" $feval > $fres
done
