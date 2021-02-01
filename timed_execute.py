import os
import time
import tqdm
import sqlite3
import joblib
from eval_scripts import evaluation
from preprocess_nl2sql import SQLDataset, ValueAlignmentException
from multiprocessing import Process, Queue


def res_map(res, val_units):
    rmap = {}
    for idx, val_unit in enumerate(val_units):
        key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
        rmap[key] = [r[idx] for r in res]
    return rmap


def execute(queue, db, p_str, p_sql=None, silent=False):
    conn = sqlite3.connect(db)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
        if p_sql is not None:
            p_val_units = [unit[1] for unit in p_sql['select'][1]]
            p_res = res_map(p_res, p_val_units)
    except Exception as e:
        conn.close()
        if not silent:
            print('failed to execute {}'.format(db))
            print(p_str)
            raise e
    else:
        conn.close()
        queue.put(p_res)


def timed_execute(db_path, query_recov, timeout=5, query_sql=None, silent=False, sleep=0.1):
    spacing = [('< =', '<='), ('> =', '<='), ('! =', '!=')]
    for f, t in spacing:
        query_recov = query_recov.replace(f, t)

    queue = Queue()
    p = Process(target=execute, args=(queue, db_path, query_recov, query_sql, silent))
    start = time.time()
    p.start()
    finished = False
    while time.time() - start <= timeout:
        if not p.is_alive():
            finished = True
            break
        time.sleep(sleep)
    else:
        p.terminate()
        p.join()
    if finished:
        try:
            result = queue.get_nowait()
        except:
            return None
    else:
        return None
    return result


def batch_execute_one(p_str, p_sql, db, silent):
    conn = sqlite3.connect(db)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
        if p_sql is not None:
            p_val_units = [unit[1] for unit in p_sql['select'][1]]
            p_res = res_map(p_res, p_val_units)
    except Exception as e:
        if not silent:
            print('failed to execute {}'.format(db))
            print(p_str)
            print(e)
        return None
    else:
        return p_res


def batch_execute(data, silent=False, sleep=0.1, timeout=5, n_proc=5, desc='batch execute'):
    spacing = [('< =', '<='), ('> =', '<='), ('! =', '!=')]
    proc = []
    for db, query, sql in data:
        for f, t in spacing:
            query_recov = query.replace(f, t)
        proc.append((query, sql, db, silent))

    par = joblib.Parallel(n_proc, backend='threading')
    out = par(joblib.delayed(batch_execute_one)(*args) for args in tqdm.tqdm(proc, desc=desc))
    return out


if __name__ == '__main__':
    db_id = 'soccer_1'
    ftables = os.path.join('data', 'spider', 'tables.json')

    db_path = 'data/database/flight_4/flight_4.sqlite'
    query = "select T1.airline from routes as T1 join airports as T2 on T1.src_apid = T2.apid where T2.city != 'Imo'"
    kmaps = evaluation.build_foreign_key_map_from_json(ftables)
    schema = evaluation.Schema(evaluation.get_schema(db_path))
    print(timed_execute(db_path, query, timeout=2))
