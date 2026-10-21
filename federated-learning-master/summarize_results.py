"""汇总已完成的同任务多种子结果；不启动训练或协议。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from utils.experiment import save_json


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_task(task_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    task_dir = Path(task_dir).resolve()
    runs, excluded, baseline_config = [], [], None
    for path in sorted(task_dir.glob('seed_*/*/summary.json')):
        directory = path.parent
        if directory.name not in ('float', 'trsfl'):
            continue
        status = read_json(directory / 'status.json')
        if status['status'] != 'complete':
            excluded.append(str(directory))
            continue
        config = read_json(directory / 'config.json')
        pair = {key: value for key, value in config.items()
                if key not in ('seed', 'initial_model_sha256', 'partition_sha256',
                               'aggregation', 'arguments', 'environment')}
        if baseline_config is None:
            baseline_config = pair
        if pair != baseline_config:
            raise ValueError("汇总目录混有不同训练/编码/代码配置，不允许混合求均值")
        rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        rows = [row for row in rows if row['status'] == 'ok']
        if [row['round'] for row in rows] != list(range(1, config['epochs'] + 1)):
            raise ValueError("标为完成的实验缺少连续轮记录: " + str(directory))
        runs.append(dict(config=config, summary=read_json(path), rows=rows))
    for status_path in task_dir.glob('seed_*/*/status.json'):
        if status_path.parent.name not in ('float', 'trsfl'):
            continue
        if read_json(status_path)['status'] != 'complete':
            excluded.append(str(status_path.parent))
    if not runs:
        raise ValueError("没有可汇总的完成结果")
    output = task_dir / 'summary'
    output.mkdir(exist_ok=True)
    final_rows, consistency_rows, statistics = [], [], {}
    for run in runs:
        config, final = run['config'], run['summary']
        final_rows.append(dict(seed=config['seed'], mode=config['aggregation'],
                               final_accuracy=final['final_accuracy'], final_loss=final['final_loss'],
                               target_accuracy=final['target_accuracy'], target_round=final['target_round'],
                               target_status=final['target_status']))
        for row in run['rows']:
            if row['mode'] == 'trsfl':
                consistency_rows.append({key: row[key] for key in (
                    'seed', 'round', 'dimension', 'participants', 'integer_sum_equal',
                    'parameter_max_abs_diff', 'parameter_mean_abs_diff',
                    'prediction_agreement', 'consistency_passed')})
    for mode in ('float', 'trsfl'):
        rows = [row for row in final_rows if row['mode'] == mode]
        if rows:
            statistics[mode] = dict(repeats=len(rows))
            for field in ('final_accuracy', 'final_loss'):
                values = [row[field] for row in rows]
                statistics[mode][field + '_mean'] = float(np.mean(values))
                statistics[mode][field + '_sample_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            reached = [row['target_round'] for row in rows if row['target_status'] == 'reached']
            statistics[mode]['target_reached_count'] = len(reached)
            statistics[mode]['target_not_reached_count'] = sum(row['target_status'] == 'not_reached' for row in rows)
            statistics[mode]['target_round_mean_among_reached'] = float(np.mean(reached)) if reached else None
    differences = []
    by_seed_mode = {(run['config']['seed'], run['config']['aggregation']): run for run in runs}
    for seed in sorted({row['seed'] for row in final_rows}):
        for left, right in (('trsfl', 'float'),):
            if (seed, left) not in by_seed_mode or (seed, right) not in by_seed_mode:
                continue
            a, b = by_seed_mode[(seed, left)], by_seed_mode[(seed, right)]
            for field in ('initial_model_sha256', 'partition_sha256'):
                if a['config'][field] != b['config'][field]:
                    raise ValueError("配对组使用了不同的初始模型或划分")
            differences.append(dict(seed=seed, comparison=left + '-' + right,
                                    final_accuracy_difference=a['summary']['final_accuracy'] - b['summary']['final_accuracy'],
                                    final_loss_difference=a['summary']['final_loss'] - b['summary']['final_loss']))
    write_csv(output / 'final_by_seed.csv', final_rows)
    write_csv(output / 'paired_differences.csv', differences)
    write_csv(output / 'consistency.csv', consistency_rows)
    save_json(output / 'statistics.json', dict(modes=statistics, excluded=sorted(set(excluded)),
                                              std_definition='sample standard deviation; null for a single repeat',
                                              missing_target_round='not reached or not configured; see target_status'))
    for field, label in (('test_accuracy', 'Test accuracy (%)'), ('test_loss', 'Test loss')):
        figure, axis = plt.subplots()
        for mode in ('float', 'trsfl'):
            selected = [run for run in runs if run['config']['aggregation'] == mode]
            if not selected:
                continue
            values = np.asarray([[row[field] for row in run['rows']] for run in selected])
            x = np.arange(1, values.shape[1] + 1)
            mean = values.mean(axis=0)
            line, = axis.plot(x, mean, label='{} (n={})'.format(mode, len(selected)))
            if len(selected) > 1:
                std = values.std(axis=0, ddof=1)
                axis.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.18)
        axis.set_xlabel('Global round')
        axis.set_ylabel(label)
        if all(statistics[mode]['repeats'] == 1 for mode in statistics):
            axis.set_title('Single run per mode; no SD band')
        else:
            axis.set_title('Mean; shading = +/- 1 sample SD when n >= 2')
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / (field + '.png'), dpi=180)
        plt.close(figure)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_dir', type=Path)
    print(summarize_task(parser.parse_args().task_dir))
